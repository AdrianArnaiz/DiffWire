from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GNNBenchmarkDataset(root='data/GNNBenchmarkDataset', name='CIFAR10')


# Make with Ahmed the features of featureless dataset
class GNNBenchmarkDatasetFeatures(GNNBenchmarkDataset):
  def __init__(self, root, name):
    super().__init__(root, name)
    self.new_num_features = 1 
    self.new_num_classes = 10
    self.new_num_graphs = 450000 
    self.data = []
    for d in dataset: 
      # Get each Data in dataset 
      adj = to_dense_adj(d.edge_index)
      A = adj
      D = A.sum(dim=1)
      x = torch.transpose(D,0,1)
      mydata = Data(x=x, edge_index=d.edge_index, y = d.y, num_nodes=d.num_nodes)
      self.data.append(mydata)
  def __getitem__(self, item):
    return self.data[item]
    

datasetGNN = GNNBenchmarkDatasetFeatures(root='data/GNNBenchmarkDataset', name='CIFAR10')
dataset = datasetGNN
data = dataset[0]  # Get the first graph object.

dataset = dataset.shuffle()

train_dataset = dataset[:40000]
test_dataset = dataset[40000:]

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True) # from 64 to 100
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)  # from 64 to 100
  
