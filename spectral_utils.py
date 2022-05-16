import torch
from torch_geometric.utils import to_dense_adj

def unnormalized_laplacian_eigenvectors(L): # 
  el, ev = torch.linalg.eig(L)
  el = torch.real(el)
  ev = torch.real(ev)
  idx = torch.argsort(el)
  el = el[idx] 
  ev = ev[:,idx]
  #print("L", L, el[1])
  return el, ev

    
# Compute Fiedler values of all the graphs
def compute_fiedler_vectors(dataset):
    """
    Calculate fieldver vector for all graphs in dataset
    """
    vectors = []
    values = []
    for g in range(len(dataset)):
        G = dataset[g]
        #print(G)
        adj = to_dense_adj(G.edge_index)
        # adj is [1, N, N]
        adj = adj.squeeze(0)
        # adj is [N, N]
        #print(adj.size())
        A = adj
        D = A.sum(dim=1)
        D = torch.diag(D)
        L = D - A
        #print(L)
        el, ev = unnormalized_laplacian_eigenvectors(L)
        vectors.append(ev[:,1])
        values.append(el[1])

    return values, vectors


