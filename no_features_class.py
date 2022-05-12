from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj


# Make with Ahmed the features of featureless dataset
class TUDatasetFeatures(TUDataset):
  def __init__(self, root, name, dataset):
    self.new_num_features = 1
    self.new_num_classes = dataset.num_classes
    self.name = name
    #self.new_num_graphs = 450000 
    self.data = []
    for d in dataset: 
      # Get each Data in dataset 
      adj = to_dense_adj(d.edge_index)
      A = adj
      D = A.sum(dim=1)
      x = torch.transpose(D,0,1)
      mydata = Data(x=x, edge_index=d.edge_index, y = d.y, num_nodes=d.num_nodes)
      self.data.append(mydata)
  @property
  def num_classes(self):
      return self.new_num_classes
  @property
  def num_features(self):
      return self.new_num_features
  def __getitem__(self, item):
    return self.data[item]
  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.name})'