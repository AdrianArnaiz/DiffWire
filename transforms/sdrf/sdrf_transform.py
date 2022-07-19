import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from transforms.sdrf.curvature import sdrf
from transforms.sdrf.utils import get_dataset

class SDRF(BaseTransform):
    
    def __init__(self,
        max_steps: int = None,
        remove_edges: bool = True,
        removal_bound: float = 0.5,
        tau: float = 1,
        undirected: bool = False,
        use_edge_weigths = True
        ):
        
        self.max_steps = int(max_steps)
        self.remove_edges = remove_edges
        self.removal_bound = removal_bound
        self.tau = tau
        self.undirected = undirected
        self.use_edge_weigths = use_edge_weigths

    def __call__(self, graph_data):

        graph_data = get_dataset(graph_data, use_lcc=False)

        altered_data = sdrf(
            graph_data,
            loops=self.max_steps,
            remove_edges=self.remove_edges,
            removal_bound=self.removal_bound,
            tau=self.tau,
            is_undirected=self.undirected,
        )

        new_data = Data(
            edge_index=torch.LongTensor(altered_data.edge_index),
            edge_attr=torch.FloatTensor(altered_data.edge_attr) if altered_data.edge_attr is not None else None,
            y=graph_data.y,
            x=graph_data.x,
            num_nodes = graph_data.num_nodes
        )      
        return new_data


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_steps})'

