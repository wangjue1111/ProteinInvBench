import time
import torch

import torch.nn as nn
import numpy as np
from src.modules.blockgat_module import *
from src.tools.affine_utils import Rigid, Rotation

class MyTokenizer:
    def __init__(self):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY' # [X] for unknown token
        self.alphabet_RNA = 'AUGC'
    
    def encode(self, seq, RNA=False):
        return [self.alphabet_protein.index(s) for s in seq]

    def decode(self, seq):
        return ''.join([self.alphabet_protein[s] for s in seq])

class BlockGAT_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(BlockGAT_Model, self).__init__()
        self.__dict__.update(locals())
        self.tokenizer = MyTokenizer()
        hidden_dim = args.hidden_dim
        geo_layer = args.geo_layer
        attn_layer = args.attn_layer
        node_layer = args.node_layer
        edge_layer = args.edge_layer
        encoder_layer = args.encoder_layer
        dropout = args.dropout
        mask_rate = args.mask_rate

        self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, 196, hidden_dim, hidden_dim)
        self.virtual_embedding = nn.Embedding(30, hidden_dim) 
        self.encoder = StructureEncoder(geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate)
        self.decoder = MLPDecoder(hidden_dim)
        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch):
        X, h_V, h_E, edge_idx, batch_id = batch['X'], batch['h_V'], batch['h_E'], batch['edge_idx'], batch['batch_id']
        T_gs, edge_idx_g, batch_id_g = self.construct_virtual_frame(batch_id, 3)

        T_ts = batch['T_ts']
        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(edge_idx.shape[1],-1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:,0].view(edge_idx_g.shape[1],-1)
        T_gs.rbf = rbf_gs
        T_ts.rbf = rbf_ts
        h_E_0 = h_E

        h_V = self.node_embedding(h_V)
        h_E = self.edge_embedding(h_E)
        num_global = (batch_id_g==0).sum()
        h_V_g = torch.arange(num_global, device=h_E.device).repeat(batch_id.unique().shape[0])
        h_V_g = self.virtual_embedding(h_V_g)
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]), device=h_V.device, dtype=h_V.dtype)
        h_E_0 = torch.cat([h_E_0, torch.zeros((edge_idx_g.shape[1], h_E_0.shape[1]), device=h_V.device, dtype=h_V.dtype)])


        h_V = self.encoder(
                                h_V, h_V_g,
                                h_E, h_E_g,
                                T_ts, T_gs, 
                                edge_idx, edge_idx_g,
                                batch_id, batch_id_g, h_E_0)
        log_probs, logits = self.decoder(h_V)

        return {'log_probs': log_probs, 'logits':logits}
    
    def _get_features(self, batch):
        X, edge_idx, batch_id = batch['X'], batch['edge_idx'], batch['batch_id']
        V, E, T, T_ts, batch_id, edge_idx, V_g, E_g, T_g, T_gs, batch_id_g, edge_idx_g = GeoFeaturizer.from_X_to_features(X, edge_idx, batch_id, merge_local_global=False)
        batch['h_V'] = V
        batch['h_E'] = E
        batch['T'] = T
        batch['T_ts'] = T_ts
        batch['E_idx'] = edge_idx
        batch['h_V_g'] = V_g
        batch['h_E_g'] = E_g
        batch['T_g'] = T_g
        batch['T_gs'] = T_gs
        batch['batch_id_g'] = batch_id_g
        batch['edge_idx_g'] = edge_idx_g
        return batch

    def construct_virtual_frame(self, batch_id, num_global=3):
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id, dim_size=batch_id.unique().shape[0])
        shift = F.pad(num_nodes, (1,0)).cumsum(dim=0)[:-1]
        global_src = torch.cat([(torch.arange(num_node, device=shift.device)+shift[i]).repeat(num_global) for i, num_node in enumerate(num_nodes)])
        global_dst = []
        for i, num_node in enumerate(num_nodes):
            global_dst.append(torch.arange(num_global, device=batch_id.device).repeat_interleave(num_node)+i*num_global+num_nodes.sum() )
                
        global_dst = torch.cat(global_dst)

        edge_idx_g = torch.cat(
            [torch.stack([global_src, global_dst]),
            torch.stack([global_dst, global_src])],
            dim=1)
        batch_id_g = torch.arange(num_nodes.shape[0], device=batch_id.device).repeat_interleave(num_global)
        
        R = torch.eye(3, device=batch_id.device)[None].repeat(edge_idx_g.shape[1],1,1)[:,None]
        trans = torch.zeros(edge_idx_g.shape[1],3, device=batch_id.device)[:,None]
        T_gs = Rigid(Rotation(R), trans)

        return T_gs, edge_idx_g, batch_id_g
    


class GeoFeaturizer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @classmethod
    @torch.no_grad()
    def from_X_to_features(self, X, edge_idx, batch_id, virtual_frame_num=3, merge_local_global=True):
        T = Rigid.make_transform_from_reference(X[:,0].view(-1,3), X[:,1].view(-1,3), X[:,2].view(-1,3))
        T_ts = T[edge_idx[1],None].invert().compose(T[edge_idx[0],None])
        V_g, T_g, T_gs, edge_idx_g, batch_id_g = self.construct_virtual_frame(T, batch_id, virtual_frame_num)

        V, E = self.get_interact_feats(T, T_ts, X, edge_idx, batch_id)
        V_g = self.int_embedding(V_g, V.shape[-1])
        E_g = torch.zeros((edge_idx_g.shape[1], E.shape[1]), device=V.device, dtype=V.dtype)

        if merge_local_global:
            V, E, T, T_ts, batch_id, edge_idx = self.merge_local_global(V, V_g, E, E_g, T, T_g, T_ts, T_gs, batch_id, edge_idx, batch_id_g, edge_idx_g)
            return V, E, T, T_ts, batch_id, edge_idx
        else:
            return V, E, T, T_ts, batch_id, edge_idx, V_g, E_g, T_g, T_gs, batch_id_g, edge_idx_g

    @classmethod
    @torch.no_grad()
    def construct_virtual_frame(self, T, batch_id, num_global=3):
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id, dim_size=batch_id.unique().shape[0])
        shift = F.pad(num_nodes, (1,0)).cumsum(dim=0)[:-1]
        # global_dst = torch.cat([batch_id + batch_id.shape[0] +k*num_nodes.shape[0] for k in range(num_global)])
        global_src = torch.cat([(torch.arange(num_node, device=shift.device)+shift[i]).repeat(num_global) for i, num_node in enumerate(num_nodes)])
        global_dst = []
        for i, num_node in enumerate(num_nodes):
            global_dst.append(torch.arange(num_global, device=batch_id.device).repeat_interleave(num_node)+i*num_global+num_nodes.sum() )
                
        global_dst = torch.cat(global_dst)

        edge_idx_g = torch.cat(
            [torch.stack([global_src, global_dst]),
            torch.stack([global_dst, global_src])],
            dim=1)
        batch_id_g = torch.arange(num_nodes.shape[0], device=batch_id.device).repeat_interleave(num_global)

        '''
        global_src: N+1,N+1,N+2,N+2,..N+B, N+B+1,N+B+1,N+B+2,N+B+2,..N+B+B
        global_dst: 0,  1,  2,  3,  ..N,   0,    1,    2,    3,    ..N
        batch_id_g: 1,  1,  2,  2,  ..B,   1,    1,    2,    2,    ..B
        '''

        X_c = T._trans
        X_m = scatter_mean(X_c, batch_id, dim=0, dim_size=batch_id.unique().shape[0])
        X_c = X_c-X_m[batch_id]
        shift = num_nodes.cumsum(dim=0)
        shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)
        all_mat = []
        for i in range(num_nodes.shape[0]):
            X_tmp = X_c[shift[i]:shift[i+1]]
            all_mat.append(X_tmp.T@X_tmp)
        U,S,V = torch.svd(torch.stack(all_mat, dim=0))
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.zeros_like(V)
        D[:, [0,1], [0,1]] = 1
        D[:,2,2] = -1*d+1*(~d)
        V = D@V
        R = torch.matmul(U, V.permute(0,2,1))
        rot_g = R.repeat_interleave(num_global,dim=0)
        trans_g = X_m.repeat_interleave(num_global,dim=0)

        T_g = Rigid(Rotation(rot_g), trans_g)
        T_all = Rigid.cat([T, T_g], dim=0)

        idx, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx,None].invert().compose(T_all[idx,None])

        h_V_g = torch.arange(num_global, device=batch_id.device).repeat(num_nodes.shape[0])
        return h_V_g, T_g, T_gs, edge_idx_g, batch_id_g

    @classmethod
    @torch.no_grad()
    def int_embedding(self, d, num_embeddings=16):
        frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        angles = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return angles

    @classmethod
    @torch.no_grad()
    def positional_embeddings(self, E_idx, dtype, num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = E_idx[0]-E_idx[1]
        E = self.int_embedding(d, num_embeddings)
        return E


    @classmethod
    @torch.no_grad()
    def get_interact_feats(self, T, T_ts, X, edge_idx, batch_id, num_rbf=16):
        dtype = X.dtype
        device = X.device
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        num_N, num_E = X.shape[0], edge_idx.shape[1]

        def rbf_func(D, num_rbf):
            shape = D.shape
            D_min, D_max, D_count = 0., 20., num_rbf
            D_mu = torch.linspace(D_min, D_max, D_count, dtype=dtype, device=device)
            D_mu = D_mu.view([1]*(len(shape))+[-1])
            D_sigma = (D_max - D_min) / D_count
            D_expand = torch.unsqueeze(D, -1)
            RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
            return RBF

        def decouple(U):
            norm = U.norm(dim=-1, keepdim=True)
            direct = U/(norm+1e-6)
            rbf = rbf_func(norm[...,0], num_rbf)
            return torch.cat([direct, rbf], dim=-1)
        
        ## ========== new_simplified_feat_diff_vec
        diffX = F.pad(X.reshape(-1,3).diff(dim=0), (0,0,1,0)).reshape(num_N, -1, 3)
        diffX_proj = T[:,None].invert()._rots.apply(diffX)
        V = decouple(diffX_proj).reshape(num_N, -1)
        V[torch.isnan(V)] = 0



        '''X [N,4,3]: N个氨基酸, 每个氨基酸4个原子(N,CA,C,O), 3是原子的xyz坐标
            T [N]: N个局部坐标系
        '''
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        
        diffE = T[src_idx,None].invert().apply(torch.cat([X[src_idx],X[dst_idx]], dim=1))
        diffE = decouple(diffE).reshape(num_E, -1)

        pos_embed = self.positional_embeddings(edge_idx, dtype, 16)
        # E = decouple(E).reshape(num_E,-1)
        E_quant = T_ts.invert()._rots._rot_mats.reshape(num_E,9)
        E_trans = T_ts._trans
        E_trans = decouple(E_trans).reshape(num_E,-1)
        E = torch.cat([diffE, E_quant, E_trans, pos_embed], dim=-1)
        return V, E

    @classmethod
    @torch.no_grad()
    def merge_local_global(self, h_V, h_V_g, h_E, h_E_g, T, T_g, T_ts, T_gs, batch_id, edge_idx, batch_id_g, edge_idx_g):
        h_V = torch.cat([h_V, h_V_g], dim=0)
        h_E = torch.cat([h_E, h_E_g], dim=0)
        T = T.cat([T, T_g], dim=0)

        batch_id = torch.cat([batch_id, batch_id_g])
        edge_idx = torch.cat([edge_idx, edge_idx_g], dim = -1)
        T_ts = T_ts.cat([T_ts, T_gs], dim = 0)
        return h_V, h_E, T, T_ts, batch_id, edge_idx
    