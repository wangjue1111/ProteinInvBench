from ast import Global
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
import numpy as np
from src.tools.design_utils import gather_nodes
import math
from src.tools.affine_utils import Rigid, quat_to_rot, rot_to_quat

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


activation_maps = {
    'leakyrelu': nn.LeakyReLU(),
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'mish': nn.Mish(),
    'gelu': nn.GELU()
}

def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

from scipy.stats import truncnorm

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        init="default",
        init_fn=None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:
                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0
                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                trunc_normal_init_(self.weight, scale=1.0)
            elif init == "relu":
                trunc_normal_init_(self.weight, scale=2.0)
            elif init == "glorot":
                nn.init.xavier_uniform_(self.weight, gain=1)
            elif init == "gating":
                with torch.no_grad():
                    self.weight.fill_(0.0)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
            elif init == "final":
                 with torch.no_grad():
                    self.weight.fill_(0.0)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        out = nn.functional.layer_norm(x, self.c_in,
            self.weight, self.bias, self.eps)
        return out


def graph2matrix(x, chunks):
    matrix = torch.stack(torch.chunk(x, chunks), dim=0)
    return matrix

class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_in, c_hidden, _outgoing=False):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_in, self.c_hidden)
        self.linear_a_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_in, self.c_hidden)
        self.linear_b_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_in, self.c_hidden, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_hidden, init="final")

        self.layer_norm_in = LayerNorm(self.c_in)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a, b):
        idxs = ((2, 0, 1), (2, 1, 0))
        idx_a, idx_b = (idxs[1], idxs[0]) if self._outgoing else (idxs[0], idxs[1])
        p = torch.matmul(
            permute_final_dims(a, idx_a),
            permute_final_dims(b, idx_b),
        )
        return permute_final_dims(p, (1, 2, 0))

    def forward(self, z, src_idx, dst_idx):
        chunks = src_idx.shape[0] // len(src_idx[src_idx == 0])
        z = graph2matrix(z, chunks)
        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        # a: [*, N_res, N_res, C_z]
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        # b: [*, N_res, N_res, C_z]
        # tri_mul_out and tri_mul_in are different here
        x = self._combine_projections(a, b)

        ndst_idx = torch.stack(torch.chunk(dst_idx, chunks), dim=0)
        ndst_idx = ndst_idx.repeat(self.c_hidden, 1, 1).permute(1, 2, 0)
        x = torch.gather(x, 1, ndst_idx)

        x = self.layer_norm_out(x)
        # x: [*, N_res, N_res, C_z]
        x = self.linear_z(x)
        # g: [*, N_res, N_res, C_z]
        g = self.sigmoid(self.linear_g(z))
        # z: [*, N_res, N_res, C_z]

        z = x * g
        return z.reshape(-1, z.shape[-1])


"""============================================================================================="""
""" Graph Encoder """
"""============================================================================================="""

def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend # 自身的mask*邻居节点的mask
    return mask_attend

#################################### node modules ###############################
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, geo_feat_dim, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp
        self.geo_feat_dim = geo_feat_dim
        
        # self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        # self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        # self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Sequential(nn.Linear(num_in+geo_feat_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden)
        )
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3+geo_feat_dim, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx=None):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update

class GCN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GCN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, src_idx, batch_id, dst_idx):
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = scatter_mean(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        return h_V

class GAT(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GAT, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, src_idx, batch_id, dst_idx):
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp()
        e = e / e.sum(-1).unsqueeze(-1)
        h_message = h_message * e.unsqueeze(-1)

        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        return h_V

class QKV(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0):
        super(QKV, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden + 12, bias=False)
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3 + 12, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        Q = self.W_Q(h_V).view(N, n_heads, 1, d)[center_id]
        K = self.W_K(h_E).view(E, n_heads, d, 1)
        attend_logits = torch.matmul(Q, K).view(E, n_heads, 1)
        attend_logits = attend_logits / np.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])

        h_V_update = self.W_O(h_V)
        return h_V_update


#################################### edge modules ###############################
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, geo_feat_dim, dropout=0.1, num_heads=None, scale=30):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.geo_feat_dim = geo_feat_dim
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
    #    self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W11 = nn.Linear( num_hidden + num_in + geo_feat_dim, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, H, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx], H], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

class DualEGraph(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DualEGraph, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        E_agg = scatter_mean(h_E, dst_idx, dim=0)
        h_EV = torch.cat([E_agg[src_idx], h_E, E_agg[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E

#################################### context modules ###############################
class Context(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_context = False, edge_context = False):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        self.V_MLP = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                )
        
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

        self.E_MLP = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden)
                                )
        
        self.E_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_V = h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
        
        # if self.edge_context:
        #     c_V = scatter_mean(h_V, batch_id, dim=0)
        #     h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return h_V, h_E


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def build_MLP(n_layers,dim_in, dim_hid, dim_out):
    layers = [nn.Linear(dim_in, dim_hid), nn.BatchNorm1d(dim_hid), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        layers.append(nn.BatchNorm1d(dim_hid))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)

class GeneralGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = 0, edge_context = 0):
        super(GeneralGNN, self).__init__()
        self.virtual_atom_num = 32
        self.geo_feat_dim = 48
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.node_net = node_net
        self.edge_net = edge_net
        if node_net == 'AttMLP':
            self.attention = NeighborAttention(num_hidden, num_in, 2*self.geo_feat_dim, num_heads=4) 
        if node_net == 'GCN':
            self.attention = GCN(num_hidden, num_in, num_heads=4) 
        if node_net == 'GAT':
            self.attention = GAT(num_hidden, num_in, num_heads=4) 
        if node_net == 'QKV':
            self.attention = QKV(num_hidden, num_in, num_heads=4) 
        
        if edge_net == 'None':
            pass
        if edge_net == 'EdgeMLP':
            self.edge_update = EdgeMLP(num_hidden, num_in, 2*self.geo_feat_dim, num_heads=4)
        if edge_net == 'DualEGraph':
            self.edge_update = DualEGraph(num_hidden, num_in, num_heads=4)
        
        self.context = Context(num_hidden, num_in, num_heads=4, node_context=node_context, edge_context=edge_context)

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        
        self.virtual_atom = nn.Linear(self.num_hidden, self.virtual_atom_num*3)
        # self.wa = nn.Linear(self.virtual_atom_num*3, self.geo_feat_dim)
        # self.wb = nn.Linear(self.virtual_atom_num*3, self.geo_feat_dim)
        # self.wa_condition = nn.Linear(self.virtual_atom_num*3+4, self.geo_feat_dim)
        # self.wb_condition = nn.Linear(self.virtual_atom_num*3+4, self.geo_feat_dim)


        # self.wa_condition = build_MLP(3, self.virtual_atom_num*3+4, num_hidden*2, self.geo_feat_dim)
        # self.wb_condition = build_MLP(3, self.virtual_atom_num*3+4, num_hidden*2, self.geo_feat_dim)


        self.w_pair = nn.Linear(self.virtual_atom_num*self.geo_feat_dim, self.geo_feat_dim*2)
        knn = 30
        self.cdconv = nn.Linear(knn*(self.virtual_atom_num+4), self.geo_feat_dim*2)

        
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, X, T, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        num_edge = src_idx.shape[0]
        

        #==================== point attention =====================
    #     VAtom = self.virtual_atom(h_V)
    #     Q = VAtom[src_idx].view(num_edge,-1,3)
    #  #   Q = Q / Q.norm(dim=-1, keepdim=True)
    #     T_ij = T[dst_idx,None].invert().compose(T[src_idx,None])
    #     quat_ij = rot_to_quat(T_ij._rots._rot_mats)
    #     dim_in = self.virtual_atom_num*3
    #     Wa_Ba = self.wa_condition(quat_ij[:,0])
    #     Wa = Wa_Ba[:, :self.virtual_atom_num*3*self.geo_feat_dim].reshape(num_edge, dim_in, self.geo_feat_dim)
    #     Ba = Wa_Ba[:, dim_in*self.geo_feat_dim:]

    #     Wb_Bb = self.wb_condition(quat_ij[:,0])
    #     Wb = Wb_Bb[:, :self.virtual_atom_num*3*self.geo_feat_dim].reshape(num_edge,dim_in, self.geo_feat_dim)
    #     Bb = Wb_Bb[:, dim_in*self.geo_feat_dim:]

    #     K = T_ij.apply(Q)
    #     # H =self.wa(Q.view(num_edge, -1)) + self.wb(K.view(num_edge, -1))
    #     H = torch.einsum('eij,ei->ej', Wa, Q.view(num_edge, -1)) + Ba + \
    #         torch.einsum('eij,ei->ej', Wb, K.view(num_edge, -1)) + Bb
    #     H_norm = H.norm(dim=-1, keepdim=True)
    #     G = torch.concat([H/H_norm, rbf(H_norm, 0, 50, self.geo_feat_dim)[:,0]], dim=-1)
        
        # #==================== point cross attention =====================
        # Q = self.virtual_atom(h_V)[src_idx].view(num_edge,-1,3)
        # T_ij = T[dst_idx,None].invert().compose(T[src_idx,None])
        # quat_ij = rot_to_quat(T_ij._rots._rot_mats)[:, 0]

        # K = T_ij.apply(Q)
        # Q_condition = self.wa_condition(torch.cat([quat_ij, Q.view(num_edge, -1)], dim = -1))
        # K_condition = self.wb_condition(torch.cat([quat_ij, K.view(num_edge, -1)], dim = -1))
        # H = Q_condition + K_condition

        # H_norm = H.norm(dim=-1, keepdim=True)
        # G = torch.concat([H/H_norm, rbf(H_norm, 0, 50, self.geo_feat_dim)[:,0]], dim=-1)


        # # #==================== pair dist attention  =====================
        # num_node = h_V.shape[0]
        # Q = self.virtual_atom(h_V).view(num_node, -1,3)
        # # Q = Q/Q.norm(dim=-1, keepdim=True)
        # Q_global = T[:,None].apply(Q)
        # D_ij = (Q_global[src_idx]-Q_global[dst_idx]).norm(dim=-1)
        # G = rbf(D_ij, 0, 50, self.geo_feat_dim).view(num_edge, -1)
        # G = self.w_pair(G)
        
        # ==================== CDConv attention  我在跑这个代码=====================
        num_node = h_V.shape[0]
        T_ij = T[dst_idx,None].invert().compose(T[src_idx,None])
        quat_ij = rot_to_quat(T_ij._rots._rot_mats)[:, 0]
        Q = self.virtual_atom(h_V).view(num_node, -1,3)
        Q_global = T[:,None].apply(Q)
        D_ij = (Q_global[src_idx]-Q_global[dst_idx]).norm(dim=-1)
        G = torch.cat([D_ij, quat_ij], dim=-1)
        knn = (src_idx==0).sum()
        G_batch = G.reshape(num_node,knn,-1) # [num_node,knn,32+4]
        G = torch.einsum('nki,kij->nkj', G_batch, self.cdconv.weight.view(knn,-1,self.geo_feat_dim*2)) # key point: 每一个knn单独使用一个可学习权重卷积对应的特征
        G = G.reshape(num_edge, -1)

        
        
        if self.node_net == 'AttMLP' or self.node_net == 'QKV':
            # =============== aggregate point-wise information into node update
            dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx], G], dim=-1), src_idx, batch_id, dst_idx)
        else:
            dh = self.attention(h_V, h_E, src_idx, batch_id, dst_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if self.edge_net=='None':
            pass
        else:
            # ============== aggregate point information into edge update
            h_E = self.edge_update( h_V, h_E, G, edge_idx, batch_id )

        # atom update
   #     X = self.atom_update(h_V).view(-1, 4, 3)

        h_V, h_E = self.context(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E, X

class GNNModule(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0):
        super(GNNModule, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads, edge_drop=0.0) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        center_id = edge_idx[0]
        dh = self.attention(h_V, h_E, center_id, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V


class GNNModule_E1(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, att_output_mlp=True, node_output_mlp=True):
        super(GNNModule_E1, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_output_mlp = node_output_mlp
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=att_output_mlp) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        # self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        # self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        # self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        if self.node_output_mlp:
            dh = self.dense(h_V)
            h_V = self.norm[1](h_V + self.dropout(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm[2](h_E + self.dropout(h_message))
        return h_V, h_E


class GNNModule_E2(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GNNModule_E2, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class GNNModule_E3(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(GNNModule_E3, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))

        from .proteinmpnn_module import PositionWiseFeedForward
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        """ Parallel computation of full transformer layer """
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp()
        e = e / e.sum(-1).unsqueeze(-1)
        h_message = h_message * e.unsqueeze(-1)

        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0, updating_edges=0, att_output_mlp=True, node_output_mlp=True, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = False, edge_context = False):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        # self.encoder_layers = nn.ModuleList([])
        encoder_layers = []
        
        self.updating_edges = updating_edges
        if updating_edges == 0:
            module = GNNModule
        elif updating_edges == 1:
            module = GNNModule_E1
        elif updating_edges == 2:
            module = GNNModule_E2
        elif updating_edges == 3:
            module = GNNModule_E3
        elif updating_edges == 4:
            module = GeneralGNN

        if updating_edges == 4:
            for i in range(num_encoder_layers):
                encoder_layers.append(
                    module(hidden_dim, hidden_dim*2, dropout=dropout, node_net = node_net, edge_net = edge_net, node_context = node_context, edge_context = edge_context),
                )
        else:
            for i in range(num_encoder_layers):
                encoder_layers.append(
                    module(hidden_dim, hidden_dim*2, dropout=dropout, att_output_mlp=att_output_mlp, node_output_mlp=node_output_mlp),
                )
        
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, h_V, h_P, X, T, P_idx, batch_id):
        for layer in self.encoder_layers:
            if self.updating_edges == 0:
                h_V = layer(h_V, torch.cat([h_P, h_V[P_idx[1]]], dim=1), T, P_idx, batch_id)
                # h_V = h_V + layer(h_V, torch.cat([h_P, h_V[P_idx[1]]], dim=1), P_idx, batch_id)
            else:
                h_V, h_P, X = layer(h_V, h_P, X, T, P_idx, batch_id)
        return h_V, h_P

"""============================================================================================="""
""" Sequence Decoder """
"""============================================================================================="""

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, padding, act_func, glu=0):
        super().__init__()

        self.glu = glu
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = act_func
        if glu == 0:
            self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        elif glu == 1:
            self.conv = nn.Conv1d(hidden_dim, 2*hidden_dim, kernel_size, padding=padding)
    
    def forward(self, x):
        if self.glu == 0:
            return self.conv(self.act(self.bn(x)))
        elif self.glu == 1:
            f_g = self.conv(self.act(self.bn(x)))
            split_dim = f_g.shape[1] // 2
            f_x, g_x = torch.split(f_g, split_dim, dim=1)
            return torch.sigmoid(g_x) * f_x

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=21):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # module_lst = [nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)]
        # for _ in range(num_layers):
        #     module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        # self.CNN = nn.Sequential(*module_lst)
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id=None, token_mask=None):
        # h_V = h_V.unsqueeze(0).permute(0,2,1)
        # hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(h_V)
        # if token_mask is not None:
        #     token_mask = token_mask[None,:].to(h_V.device)
        #     logits = logits*token_mask -999999*(~token_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=20):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        module_lst = [nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)]
        for _ in range(num_layers):
            module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        self.CNN = nn.Sequential(*module_lst)

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id):
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class CNNDecoder2(nn.Module):
    def __init__(self,hidden_dim, input_dim, num_layers=3, kernel_size=5, act_type='relu', glu=0, vocab=20):
        super().__init__()
        self.ConfNN = nn.Embedding(50, hidden_dim)

        padding = (kernel_size - 1) // 2
        module_lst = [nn.Conv1d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=padding)]
        for _ in range(num_layers):
            module_lst.append(ConvBlock(hidden_dim, kernel_size, padding, activation_maps[act_type], glu))

        self.CNN = nn.Sequential(*module_lst)

        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, logits, batch_id):
        eps = 1e-5
        L = h_V.shape[0]
        idx = torch.argsort(-logits, dim=1)
        Conf = logits[range(L), idx[:,0]] / (logits[range(L), idx[:,1]] + eps)
        Conf = Conf.long()
        Conf = torch.clamp(Conf, 0, 49)
        h_C = self.ConfNN(Conf)
        
        h_V = torch.cat([h_V,h_C],dim=-1)
        h_V = h_V.unsqueeze(0).permute(0,2,1)
        hidden = self.CNN(h_V).permute(0,2,1).squeeze()
        logits = self.readout(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h

class Local_Module(nn.Module):
    def __init__(self, num_hidden, num_in, is_attention, dropout=0.1, scale=30):
        super(Local_Module, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.is_attention = is_attention
        self.scale = scale
        self.dropout = nn.Dropout(0)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.W = nn.Sequential(*[
            nn.Linear(num_hidden + num_in, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden)
        ])
        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx):
        message = torch.cat( [h_V[edge_idx[0]], h_E], dim=1 )
        h_message = self.W(message) # [17790, 128]
        # Attention
        if self.is_attention == 1:
            att = F.sigmoid(F.leaky_relu(torch.matmul(message, self.A))).exp()
            att = att / scatter_sum(att, edge_idx[0], dim=0)[edge_idx[0]]
            h_message = h_message * att # [4, 312, 30, 128]

        # message aggragation
        dh = scatter_sum(h_message, edge_idx[0], dim=0) / self.scale
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V
    





    # def forward(self, S, h_V, h_E, edge_idx, batch_id):
    #     h_S = self.W_s(S)
    #     known_mask = (edge_idx[0]>edge_idx[1]).unsqueeze(-1)
    #     h_ES_known = self.UpdateE(torch.cat([h_E, h_S[edge_idx[1]]], dim=-1))

    #     for dec_layer in self.decoder:
    #         h_E_mix = h_ES_known*known_mask + h_E*(~known_mask)
    #         h_V, _ = dec_layer(h_V, h_E_mix, edge_idx, batch_id)
        
    #     log_probs, logits = self.readout(h_V)
    #     return log_probs
    
    # def sampling(self, h_V, h_E, edge_idx, batch_id, temperature=0.1):
    #     device = h_V.device
    #     L = h_V.shape[0]

    #     # cache
    #     S = torch.zeros( L, device=device, dtype=torch.int)
    #     h_S = torch.zeros( L, self.hidden_dim, device=device)
    #     h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder))]
    #     log_probs = torch.zeros( L, self.vocab, device=device)
        
    #     for t in range(L):
    #         edge_mask = edge_idx[0] % L == t # 批量预测第t个氨基酸
    #         h_V_t = h_V[t:t+1,:]
    #         E_idx_t = edge_idx[:, edge_mask]
    #         h_ES_known = torch.cat((h_E, h_S[edge_idx[1]]), dim=1)
    #         h_ES_known_t = self.UpdateE(h_ES_known[edge_mask])
    #         h_E_t = h_E[edge_mask]
    #         known_mask = (E_idx_t[0]>E_idx_t[1]).unsqueeze(-1)
        
    #         for l, dec_layer in enumerate(self.decoder):
    #             h_ES_t = h_ES_known_t*known_mask + h_E_t*(~known_mask)
                
    #             h_V_t = h_V_stack[l][E_idx_t[1],:]
    #             edge_index_t_local = torch.zeros_like(E_idx_t)
    #             edge_index_t_local[1,:] = torch.arange(0, E_idx_t.shape[1], device=h_V.device)
    #             batch_id_t = torch.zeros(h_V_t.shape[0], dtype=torch.long, device=device)
    #             h_V_t, _ = dec_layer(h_V_t, h_ES_t , edge_index_t_local, batch_id_t)
                
    #             h_V_stack[l+1][t] = h_V_t[0]
            
    #         h_V_t = h_V_stack[-1][t]
    #         log_probs_t, logits_t = self.readout(h_V_t) 
    #         log_probs[t] = log_probs_t
    #         probs = F.softmax(logits_t/temperature, dim=-1)
    #         S_t = torch.multinomial(probs, 1).squeeze(-1)
    #         h_S[t::L] = self.W_s(S_t)
    #         S[t::L] = S_t
        
    #     return log_probs

if __name__ == '__main__':
    pass