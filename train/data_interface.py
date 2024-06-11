import inspect
from torch.utils.data import DataLoader
from src.interface.data_interface import DInterface_base
import torch
import os.path as osp
from src.tools.utils import cuda
import copy
import random
import torch.utils.data as data

class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
        self.batch_size = 8
        self.drop_last = False
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: 
            self._form_batches()
        for batch in self.batches: 
            yield batch

class MyDataLoader(DataLoader):
    def __init__(self, dataset, model_name, batch_size=64, num_workers=8, *args, **kwargs):
        super(MyDataLoader, self).__init__(dataset, batch_size, num_workers=num_workers, *args, **kwargs)
        self.pretrain_device = 'cuda:0'
        self.model_name = model_name
    
    def __iter__(self):
        for batch in super().__iter__():
            # 在这里对batch进行处理
            # ...
            try:
                self.pretrain_device = f'cuda:{torch.distributed.get_rank()}'
            except:
                self.pretrain_device = 'cuda:0'

            stream = torch.cuda.Stream(
                self.pretrain_device
            )
            with torch.cuda.stream(stream):
                if self.model_name=='GVP':
                    batch = batch.cuda(non_blocking=True, device=self.pretrain_device)
                    yield batch
                else:
                    for key, val in batch.items():
                        if type(val) == torch.Tensor:
                            batch[key] = batch[key].cuda(non_blocking=True, device=self.pretrain_device)

                    # X = batch['X'].cuda(non_blocking=True, device=self.pretrain_device)
                    # S = batch['S'].cuda(non_blocking=True, device=self.pretrain_device)
                    # score = batch['score'].cuda(non_blocking=True, device=self.pretrain_device)
                    # mask = batch['mask'].cuda(non_blocking=True, device=self.pretrain_device)
                    # lengths = batch['lengths'].cuda(non_blocking=True, device=self.pretrain_device)
                    # chain_mask = batch['chain_mask'].cuda(non_blocking=True, device=self.pretrain_device)
                    # chain_encoding = batch['chain_encoding'].cuda(non_blocking=True, device=self.pretrain_device)
                
                    yield batch


class DInterface(DInterface_base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.load_data_module()
        self.dataset = self.instancialize(split = 'train')

    def setup(self, stage=None):
        from src.datasets.featurizer import (featurize_AF, featurize_GTrans, featurize_GVP, featurize_ProteinMPNN, featurize_Inversefolding, blockgat_collate_fn)
        if self.hparams.model_name in ['AlphaDesign', 'PiFold', 'PiFold-wj', 'KWDesign', 'GraphTrans', 'StructGNN', 'GCA', 'E3PiFold']:
            self.collate_fn = featurize_GTrans
        elif self.hparams.model_name == 'GVP':
            featurizer = featurize_GVP()
            self.collate_fn = featurizer.collate
        elif self.hparams.model_name == 'ProteinMPNN':
            self.collate_fn = featurize_ProteinMPNN
        elif self.hparams.model_name == 'ESMIF':
            self.collate_fn = featurize_Inversefolding
        elif self.hparams.model_name == 'BlockGAT':
            self.collate_fn = blockgat_collate_fn
        elif self.hparams.model_name == 'KWDesign_BlockGAT':
            self.collate_fn = blockgat_collate_fn
    
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = copy.deepcopy(self.dataset)
            self.trainset.change_mode('train')
            self.valset = copy.deepcopy(self.dataset)
            self.valset.change_mode('valid')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = copy.deepcopy(self.dataset)
            self.testset.change_mode('test')

    def train_dataloader(self):
        return MyDataLoader(self.trainset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=True, prefetch_factor=8, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):

        return MyDataLoader(self.valset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return MyDataLoader(self.testset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    def load_data_module(self):
        name = self.hparams.dataset
        if name == 'AF2DB':
            from src.datasets.alphafold_dataset import AlphaFoldDataset
            self.data_module = AlphaFoldDataset
        
        if name == 'TS':
            from src.datasets.ts_dataset  import TSDataset
            self.data_module = TSDataset
            self.hparams['path'] = osp.join(self.hparams.data_root, 'ts')
        
        if name == 'CASP15':
            from src.datasets.casp_dataset  import CASPDataset
            self.data_module = CASPDataset
            self.hparams['path'] = osp.join(self.hparams.data_root, 'casp15')
        
        if name == 'CATH4.2':
            from src.datasets.cath_dataset import CATHDataset
            self.data_module = CATHDataset
            self.hparams['version'] = 4.2
            self.hparams['path'] = osp.join(self.hparams.data_root, 'cath4.2')
            
        if name == 'CATH4.3':
            from src.datasets.cath_dataset import CATHDataset
            self.data_module = CATHDataset
            self.hparams['version'] = 4.3
            self.hparams['path'] = osp.join(self.hparams.data_root, 'cath4.3')
        
        if name == 'MPNN':
            from src.datasets.mpnn_dataset import MPNNDataset
            self.data_module = MPNNDataset

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        
        class_args =  list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)