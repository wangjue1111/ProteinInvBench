import os
import json
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
import numpy as np
from src.tools.affine_utils import Rotation, Rigid, quat_to_rot, rot_to_quat, invert_rot_mat
from torch_geometric.nn.pool import knn_graph
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTokenizer:
    def __init__(self):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY' # [X] for unknown token
        self.alphabet_RNA = 'AUGC'
    
    def encode(self, seq, RNA=False):
        if RNA:
            return [self.alphabet_RNA.index(s) for s in seq]
        else:
            return [self.alphabet_protein.index(s) for s in seq]

class CATHDataset(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2, k_neighbors=30):
        self.__dict__.update(locals())
        
        if data is None:
            self.data = self.cache_data[split]
        else:
            self.data = data
        
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        # print(self.tokenizer.mask_token_id)
        self.tokenizer = MyTokenizer()
    
    
    def read_line(self, line, alphabet_set):
        entry = json.loads(line)
        seq = entry['seq']

        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)

        bad_chars = set([s for s in seq]).difference(alphabet_set)

        if len(bad_chars) == 0:
            if len(entry['seq']) <= self.max_length:
                chain_length = len(entry['seq'])
                chain_mask = np.ones(chain_length)
                return {
                    'title':entry['name'],
                    'type':1,
                    'seq':entry['seq'],
                    'CA':entry['coords']['CA'],
                    'C':entry['coords']['C'],
                    'O':entry['coords']['O'],
                    'N':entry['coords']['N'],
                    'chain_mask': chain_mask,
                    'chain_encoding': 1*chain_mask
                }
            
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise "no such file:{} !!!".format(self.path)
        else:
            with open(self.path+'/chain_set.jsonl') as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
                        
            if self.version==4.2:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.version==4.3:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            
            if self.test_name == 'L100':
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc':
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            data_dict = {'train':[],'valid':[],'test':[]}
            for data in data_list:
                if name2set.get(data['title']):
                    if name2set[data['title']] == 'train':
                        data_dict['train'].append(data)
                    
                    if name2set[data['title']] == 'valid':
                        data_dict['valid'].append(data)
                    
                    if name2set[data['title']] == 'test':
                        data['category'] = 'Unkown'
                        data['score'] = 100.0
                        data_dict['test'].append(data)
            return data_dict

    # @cached_property
    # def cache_data(self):

    #     data_dict = {'train': [], 'val': [], 'test': []}
    #     alphabet='ACDEFGHIKLMNPQRSTVWY'
    #     alphabet_set = set([a for a in alphabet])

    #     self.path = 'data/cath4.3'
    #     if not os.path.exists(self.path):
    #         raise "no such file:{} !!!".format(self.path)
    #     else:
    #         with open(self.path+'/chain_set.jsonl') as f:
    #             lines = f.readlines()

    #         data_list = []
    #         for line in tqdm(lines):
    #             data = self.read_line(line, alphabet_set)
    #             if data:
    #                 data_list.append(data)

    #         with open(self.path+'/chain_set_splits.json') as f:
    #             dataset_splits = json.load(f)
    #         name2set = {}
    #         name2set.update({name:'train' for name in dataset_splits['train']})
    #         name2set.update({name:'valid' for name in dataset_splits['validation']})
    #         name2set.update({name:'test' for name in dataset_splits['test']})
    #         for data in data_list:
    #             if name2set.get(data['title']):
    #                 if name2set[data['title']] == 'train':
    #                     data_dict['train'].append(data)

    #                 if name2set[data['title']] == 'valid':
    #                     data_dict['val'].append(data)

    #                 if name2set[data['title']] == 'test':
    #                     data['category'] = 'Unkown'
    #                     data['score'] = 100.0
    #                     data_dict['test'].append(data)

    #     #========================Protein data===============================#
    #     self.path = 'data/novelseq/pred_novelseq.jsonl'
    #     data_list = []
    #     if not os.path.exists(self.path):
    #         raise "no such file:{} !!!".format(self.path)
    #     else:
    #         with open(self.path) as f:
    #             lines = f.readlines()
    #         for line in tqdm(lines):
    #             data = self.read_line(line, alphabet_set)
    #             if data:
    #                 data_list.append(data)
    #         data_dict['test'] = data_list

    #         return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def _get_features(self, batch):
        S,  X = batch['S'], batch['X']

        X, S = X.unsqueeze(0), S.unsqueeze(0)
        mask = torch.isfinite(torch.sum(X,(2,3))).float() # atom mask
        numbers = torch.sum(mask, axis=1).int()
        S_new = torch.zeros_like(S)
        X_new = torch.zeros_like(X)+torch.nan
        for i, n in enumerate(numbers):
            X_new[i,:n,::] = X[i][mask[i]==1]
            S_new[i,:n] = S[i][mask[i]==1]

        X = X_new
        S = S_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X,(2,3))).float()
        X[isnan] = 0.

        mask_bool = (mask==1)
        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], shape[1],-1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1,*shape[2:])
            return out

        batch_id = torch.arange(mask_bool.shape[0], device=mask_bool.device)[:,None].expand_as(mask_bool)

        seq = node_mask_select(S)
        X = node_mask_select(X)
        batch_id = node_mask_select(batch_id)
        mask = torch.masked_select(mask, mask_bool)

        C_a = X[:,1,:]
        edge_idx = knn_graph(C_a, k=self.k_neighbors, batch=batch_id, loop=True, flow='target_to_source')
        batch_update={
                'X':X,
                'S': seq,
                'edge_idx':edge_idx,
                'batch_id':batch_id,
                'mask':mask,
                'num_nodes':torch.tensor(X.shape[0]).reshape(1,)}
        batch.update(batch_update)
        return batch
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])

        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        item['X'] = torch.from_numpy(np.stack([item['N'], item['CA'], item['C'], item['O']], axis=1)).float()
        item['S'] = torch.tensor(self.tokenizer.encode(item['seq']))

        return self._get_features(item)