import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected
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
    def __init__(self, alpha:float, use_edge_weigths = False):
        self.alpha = alpha
        self.eps = 0.005
        self.use_edge_weigths = use_edge_weigths

    def __call__(self, data):
        new_edges, new_weights = self.digl_edges(data.edge_index, data.num_edges) 
        data.edge_index = new_edges

        if self.use_edge_weigths:
            data.edge_weight = new_weights
        
        return data
    
    
    def gdc(self, A: sp.csr_matrix, alpha: float, num_previous_edges):
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
        
        # Same as e-threshold based on average degree
        # but dynamic for all graphs
        A = np.array(S.todense())
        top_k_idx = np.unravel_index(np.argsort(A.ravel())[-num_previous_edges:], A.shape)
        mask = np.ones(A.shape, bool)
        mask[top_k_idx] = False
        A[mask] = 0
        S_tilde = sp.csr_matrix(A)


        # Column-normalized transition matrix on graph S_tilde
        D_tilde_vec = S_tilde.sum(0).A1
        T_S = S_tilde / D_tilde_vec

        return T_S

    def get_top_k_matrix(self, A: np.ndarray, k: int = 128) -> np.ndarray:
        """
        Get k best edges for EACH NODE
        """
        num_nodes = A.shape[0]
        print('AA', num_nodes)
        row_idx = np.arange(num_nodes)
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        return A/norm

    def digl_edges(self, edges, num_previous_edges):
        A0 = sp.csr_matrix(to_scipy_sparse_matrix(edges))
        new_sp_matrix = sp.csr_matrix(self.gdc(A0, self.alpha, num_previous_edges))
        new_edge_index, weights = from_scipy_sparse_matrix(new_sp_matrix)
        return new_edge_index, weights
    
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, eps={self.alpha})'


class KNNGraph(BaseTransform):
    r"""Creates a k-NN graph based on node positions :obj:`pos`
    (functional name: :obj:`knn_graph`).

    Args:
        k (int, optional): The number of neighbors. (default: :obj:`6`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        force_undirected (bool, optional): If set to :obj:`True`, new edges
            will be undirected. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`).
            If set to :obj:`"source_to_target"`, every target node will have
            exactly :math:`k` source nodes pointing to it.
            (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        k=None,
        loop=False,
        force_undirected=True,
        flow='source_to_target',
        cosine: bool = False,
        num_workers: int = 1,
    ):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine
        self.num_workers = num_workers

    def __call__(self, data):
        had_features = True
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None

        if data.x is None:
            idx = data.edge_index[1]
            deg = degree(idx, data.num_nodes, dtype=torch.float).unsqueeze(-1)
            data.x = deg
            had_features = False

        if self.k is None:
            self.k = int(data.num_edges / (data.num_nodes*4)) #mean degree - Note: num_edges is already doubled by default in PyG
        

        edge_index = torch_geometric.nn.knn_graph(
            data.x,
            self.k,
            batch,
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
            num_workers=self.num_workers,
        )
        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index

        #Update degree
        if not had_features:
            idx = data.edge_index[1]
            deg = degree(idx, data.num_nodes, dtype=torch.float).unsqueeze(-1)
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'