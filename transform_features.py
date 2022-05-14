import torch
import torch.nn.functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix

import scipy.sparse as sp
import numpy as np


class FeatureDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(self, in_degree=False, cat=True):
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.float).unsqueeze(-1)
        #deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_degree})'


class DIGLedges(BaseTransform):
    def __init__(self, alpha:float, eps:float, use_edge_weigths = False):
        self.alpha = alpha
        self.eps = eps
        self.use_edge_weigths = use_edge_weigths

    def __call__(self, data):
        new_edges, new_weights = self.digl_edges(data.edge_index, self.alpha, self.eps)
        data.edge_index = new_edges
        
        if self.use_edge_weigths:
            data.edge_weight = new_weights
            
        return data
    
    
    def gdc(self, A: sp.csr_matrix, alpha: float, eps: float):
        N = A.shape[0]

        # Self-loops
        A_loop = sp.eye(N) + A

        # Symmetric transition matrix
        D_loop_vec = A_loop.sum(0).A1
        D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
        D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
        T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

        # PPR-based diffusion
        S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

        # Sparsify using threshold epsilon
        S_tilde = S.multiply(S >= eps)

        # Column-normalized transition matrix on graph S_tilde
        D_tilde_vec = S_tilde.sum(0).A1
        T_S = S_tilde / D_tilde_vec

        return T_S

    def digl_edges(self, edges, alpha, eps):
        A0 = sp.csr_matrix(to_scipy_sparse_matrix(edges))
        new_sp_matrix = sp.csr_matrix(self.gdc(A0, self.alpha, self.eps))
        new_edge_index, weights = from_scipy_sparse_matrix(new_sp_matrix)
        return new_edge_index, weights
    
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha})'