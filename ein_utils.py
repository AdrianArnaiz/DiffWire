import torch
#  Trace of a tensor [1,k,k]
def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

# Diagonal version of a tensor [1,n] -> [1,n,n]
def _rank3_diag(x):
    # Eye matrix of n=x.size(1): [n,n]
    eye = torch.eye(x.size(1)).type_as(x)
    #print(eye.size())
    #print(x.unsqueeze(2).size())
    # x.unsqueeze(2) adds a second dimension to [1,n] -> [1,n,1]
    # expand(*x.size(), x.size(1)) takes [1,n,1] and expands [1,n] with n -> [1,n,n]
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1)) 
    return out