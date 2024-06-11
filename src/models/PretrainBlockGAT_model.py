import torch
import os.path as osp
from src.models.blockgat_model import BlockGAT_Model
import torch.nn.functional as F
from torch_scatter import scatter_sum
from src.tools.affine_utils import Rigid, Rotation
from transformers import AutoTokenizer

def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)

class PretrainBlockGAT_Model(BlockGAT_Model):
    def __init__(self, args, pretrain_pifold_path, **kwargs):
        """ Graph labeling network """
        super().__init__(args)
        params = torch.load(pretrain_pifold_path)
        params2 = {}
        for key, val in params.items():
            params2[key.replace('_forward_module.model.', '')] = val
        self.load_state_dict(params2)
        self.esm_tokenizer =  AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
    
    @torch.no_grad()
    def forward(self, batch):
        batch_id = batch['batch_id']
        h_V, h_E, edge_idx, batch_id = batch['h_V'], batch['h_E'], batch['E_idx'], batch['batch_id']
        T_ts = batch['T_ts']
        T_gs, edge_idx_g, batch_id_g = self.construct_virtual_frame(batch_id, 3)
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
        h_S = None
        h_E_0 = torch.cat([h_E_0, torch.zeros((edge_idx_g.shape[1], h_E_0.shape[1]), device=h_V.device, dtype=h_V.dtype)])


        h_V = self.encoder(h_V, h_V_g,
                                h_E, h_E_g,
                                T_ts, T_gs, 
                                edge_idx, edge_idx_g,
                                batch_id, batch_id_g, h_E_0)
        log_probs, logits_old = self.decoder(h_V)

        # 按照esmtokenizer重排logits
        reidx = [self.esm_tokenizer._token_to_id[s] for s in self.tokenizer.alphabet_protein]
        logits = torch.zeros_like(logits_old)-1000
        logits[:, reidx] = logits_old[:, torch.arange(20)]

        probs = F.softmax(logits, dim=-1)
        conf, pred_id = probs.max(dim=-1)
        device = conf.device
        
        maxL = 0
        for b in batch_id.unique():
            mask = batch_id==b
            L = mask.sum()
            if L>maxL:
                maxL=L
        
        confs = []
        seqs = []
        embeds = []
        probs2 = []
        for b in batch_id.unique():
            mask = batch_id==b
            elements = self.esm_tokenizer.decode(pred_id[mask]).split(" ")
            seqs.append(elements)
            confs.append(conf[mask])
            embeds.append(h_V[mask])
            probs2.append(probs[mask])
        
        seqs = self.esm_tokenizer(["".join(one) for one in seqs], padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
        confs = torch.stack([F.pad(one, (0, maxL-len(one))) for one in confs])
        embeds = torch.stack([F.pad(one, (0,0, 0, maxL-len(one))) for one in embeds])
        probs2 = torch.stack([F.pad(one, (0,0, 0, maxL-len(one)), value=1/33) for one in probs2])
        
        ret = {"pred_ids":seqs['input_ids'].to(device),
               "confs":confs,
               "embeds":embeds,
               "probs":probs2,
               "attention_mask":seqs['attention_mask'].to(device),
               "E_idx":batch['E_idx'],
               "batch_id":batch_id,
               "h_E":h_E}
        return ret

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
