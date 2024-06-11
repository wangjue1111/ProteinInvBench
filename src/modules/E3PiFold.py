import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from src.modules.invariant_point_attention import InvariantPointAttention
from omegaconf import OmegaConf

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = F.gelu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class GaussianLayer(nn.Module):
    def __init__(self, num_distance=25, K=16, edge_dim=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, num_distance*self.K)  # 16 * 25 = 400, it's the total number of kernels
        self.stds = nn.Embedding(1, num_distance*self.K)
        self.mul = nn.Linear(edge_dim, num_distance)
        self.bias = nn.Linear(edge_dim, num_distance)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_feat):
        mul = self.mul(edge_feat).type_as(x)
        bias = self.bias(edge_feat).type_as(x)

        x = mul * x + bias # [B, N, N, 25]
        x = x.unsqueeze(-1) # [B, N, N, 25, 1]
        x = x.expand(-1, -1, -1, -1, self.K) # [B, N, N, 25, K]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1) # [B, N, N, 25*K]
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class GaussianEncoder(nn.Module):
    def __init__(self, kernel_num, feat_dim, num_head, use_dist=1, use_product=1):
        super().__init__()
        self.num_distance = 0
        self.use_dist = use_dist
        self.use_product = use_product
        if use_dist:
            self.num_distance += 1

        if use_product:
            self.num_distance += 1

        self.gbf = GaussianLayer(self.num_distance, kernel_num, feat_dim)
        self.node_gate = nn.Linear(feat_dim, 1)

        self.gbf_proj = NonLinearHead(
            input_dim=kernel_num*self.num_distance,
            out_dim=num_head,
            hidden=128,
        )
        self.centrality_proj = NonLinearHead(
            input_dim=kernel_num*self.num_distance,
            out_dim=feat_dim,
            hidden=1024,
        )
    
    def get_encoding_features(self, dist, et, pair_mask=None, get_bias=True):
        n_node = dist.size(-2)
        gbf_feature = self.gbf(dist, et)

        if pair_mask is not None:
            centrality_encoding = gbf_feature * pair_mask.unsqueeze(-1)
        else:
            centrality_encoding = gbf_feature # [B, N, N, 25*K]
        centrality_encoding = self.centrality_proj(centrality_encoding.sum(dim=-2)) # [B, N, encoder_embed_dim]

        graph_attn_bias = self.gbf_proj(gbf_feature) # [B, N, N, num_head]
        if get_bias:
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() # [B, num_head, N, N]
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node) # [B*num_head, N, N]
        return graph_attn_bias, centrality_encoding

    
    def build_pairwise_product_dist(self, coords, node_feat):
        dist = _get_dist(coords,coords)
        coords = coords * self.node_gate(node_feat)
        pretext = coords[:,:,None]+coords[:,None,:]
        A = torch.einsum('bijd,bjd->bij', pretext, coords)
        B = torch.einsum('bid,bjd->bij', coords, coords)
        product = A*B
        product, dist = product[...,None], dist[...,None]
        geo_feat = torch.empty_like(product)[...,0:0]
        if self.use_dist:
            geo_feat = torch.cat([geo_feat, dist], dim=-1)
        
        if self.use_product:
            geo_feat = torch.cat([geo_feat, product], dim=-1)
        return geo_feat
    
    def forward(self, coords, node_feat, pair_mask=None, get_bias=True):
        geo_feat = self.build_pairwise_product_dist(coords, node_feat)
        edge_feat = node_feat[:,:,None,:]-node_feat[:,None,:,:]
        graph_attn_bias, centrality_encoding = self.get_encoding_features(geo_feat, edge_feat, pair_mask=pair_mask, get_bias=get_bias)
        x = centrality_encoding
        return x, graph_attn_bias

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        else:
            attn_weights += attn_bias
            attn = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        # pdb.set_trace()
        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = F.gelu
        # self.edge_attn_hidden_dim = edge_attn_hidden_dim
        # self.edge_attn_heads = edge_attn_heads

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        c_s: int = 768,
        c_z: int = 16,
        attention_heads: int = 8,
    ) -> None:

        super().__init__()
        self.embed_dim = c_s
        self.attention_heads = attention_heads
        self.input_linear = nn.Linear(4*3, c_s)
        self.pred_linear = nn.Linear(c_s, 33)

        self.layers = nn.ModuleList(
            [
                InvariantPointAttention(
                    c_s=self.embed_dim,
                    c_z=c_z,
                    c_hidden=32,
                    no_heads=attention_heads,
                    no_qk_points = 30,
                    no_v_points = 30,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        X,
        T,
        Z,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, K = X.shape[:3]
        X_local = T[:,:,None].invert_apply(X)
        # X_local = X_local.to(X.dtype)
        X_local = self.input_linear(X_local.reshape(B,L, K*3))

        for layer in self.layers:
            X_local = layer(
                X_local,
                Z,
                T,
                mask=padding_mask
            )

        logits = self.pred_linear(X_local)
        return logits

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

def _get_dist(A, B):
    D_A_B = torch.sqrt(torch.sum((A[..., None,:] - B[...,None,:,:])**2,-1) + 1e-6) #[B, L, L]
    return D_A_B



