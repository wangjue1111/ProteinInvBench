import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np
import glob
from transformers import AutoTokenizer
from utils import Protein
from residue_constants import *

import torch.nn.functional as F
LOCAL_DATA_DIR = Path(
    "/huyuqi/dataset/AF2DB/filtered_1M"
)

# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = torch.matmul(P.transpose(1,2), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    # if d:
    #     S[-1] = -S[-1]
    #     V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = torch.matmul(V, W)

    return U


# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    a_mean = torch.mean(a, dim=1).view(-1,1,3)
    b_mean = torch.mean(b, dim=1).view(-1,1,3)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    a_aligned = torch.matmul(a_c, rotation)
    t = b_mean - torch.mean(a_aligned, dim=1).view(-1,1,3)
    a_aligned += t

    return a_aligned, rotation, t


class Af2dbDataset(Dataset):
    """
    Load in the dataset.

    All angles should be given between [-pi, pi]
    """

    feature_names = {
        "angles": [
            "0C:1N",
            "N:CA",
            "CA:C",
            "phi",
            "psi",
            "omega",
            "tau",
            "CA:C:1N",
            "C:1N:1CA",
        ],
        "coords": ["x", "y", "z"],
    }

    def __init__(
        self,
        pad: int = 512,
        min_length: int = 40,  # Set to 0 to disable
        split='train',
        noise=0.0,
        crop_seq = True,
        transform=None
    ) -> None:
        super().__init__()
        self.crop_seq = crop_seq
        self.pad = pad
        self.min_length = min_length
        self.rng = np.random.default_rng(seed=6489)
        self.fnames = glob.glob(os.path.join(LOCAL_DATA_DIR, "*"))
        self.fnames = [one for one in self.fnames if one[-3:]=='.gz']
        if split!='train':
            self.fnames = self.fnames[:1000]
        self.ESM_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        self.noise = noise
        # self.crop_pad_data = CropPadData(pad, self.ESM_tokenizer)
        self.avg_template = rigid_group_atom_positions
    
    def __len__(self):
        return len(self.fnames)
    
    def _read_protein(self, fname):
        protein = Protein(fname, device='cuda')
        X, C, S = protein.to_XCS()
        return S, X

    def _transforms(self, retval):
        pass 

    def _get_template(self):
        avg_template = torch.zeros([4,3]).cuda()
        maps = {'CA':0, 'C':1, 'N':2, 'O': 3}
        for template in templates:
            for atom in templates[template]:
                if atom[0] in maps:
                    coords = torch.Tensor(atom[2]).cuda()
                    avg_template[maps[atom[0]]] += coords/len(templates) 
        avg_templates = torch.cat([avg_template]*X.shape[1], dim = 0)
        return avg_template

    def __getitem__(
        self, index
    ):  
        fname = self.fnames[index]
        seqs, coords = self._read_protein(fname)
        template = self._get_template()
        template = torch.cat([template]*coords.shape[1], dim = 0)
        coords = kabsch(template.view(-1,4,3, coords.view(-1,4,3)))

        seqs = self.ESM_tokenizer.encode("".join(seqs), add_special_tokens=False)
        seqs = torch.from_numpy(np.array(seqs).reshape(-1,1))
        coords = torch.from_numpy(coords).float()

        retval = {
            "key": fname.split('/')[-1],
            "seq": seqs.long(),
            "coords": coords.float(),
        }
        
        retval = self.transforms(retval)
        return retval



if __name__ == '__main__':
    protein = Protein('src/datasets/AF-A0A2X2UIA8-F1-model_v4.cif', device='cuda')
    X, C, S = protein.to_XCS()
    templates = rigid_group_atom_positions
    avg_template = torch.zeros([4,3]).cuda()
    maps = {'N':0, 'CA':1, 'C':2, 'O': 3}
    for template in templates:
        for atom in templates[template]:
            if atom[0] in maps:
                coords = torch.Tensor(atom[2]).cuda()
                avg_template[maps[atom[0]]] += coords/len(templates) 
    avg_templates = torch.cat([avg_template]*X.shape[1], dim = 0)
    a = kabsch(avg_templates.view(-1,4,3), X.view(-1,4,3))
    
    b = Protein.from_XCS(a[0].view(1,100,4,3),C,S)
    b.to_CIF('src/datasets/AF.cif')

