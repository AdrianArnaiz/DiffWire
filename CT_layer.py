# Commute Times rewiring
#Graph Convolutional Network layer where the graph structure is given by an adjacency matrix. 
# We recommend user to use this module when applying graph convolution on dense graphs.
#from torch_geometric.nn import GCNConv, DenseGraphConv
import torch
from ein_utils import _rank3_diag, _rank3_trace

def dense_CT_rewiring(x, adj, s, mask=None, EPS=1e-15): # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20
    #print("Input x size to mincut pool", x.size())
    x = x.unsqueeze(0) if x.dim() == 2 else x # x torch.Size([20, 40, 32]) if x has not 2 parameters 
    #print("Unsqueezed x size to mincut pool", x.size(), x.dim()) # x.dim() is usually 3

    # adj torch.Size([20, N, N]) N=Mmax
    #print("Input adj size to mincut pool", adj.size())
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj # adj torch.Size([20, N, N]) N=Mmax
    #print("Unsqueezed adj size to mincut pool", adj.size(), adj.dim()) # adj.dim() is usually 3

    # s torch.Size([20, N, k])
    s = s.unsqueeze(0) if s.dim() == 2 else s # s torch.Size([20, N, k])
    #print("Unsqueezed s size", s.size())

    # x torch.Size([20, N, 32]) if x has not 2 parameters
    (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    #print("batch_size and num_nodes", batch_size, num_nodes, k) # batch_size = 20, num_nodes = N, k = 16
    s = torch.tanh(s) # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())

    if mask is not None: # NOT None for now
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        #print("mask size", mask.size()) # [20, N, 1]
        # Mask pointwise product. Since x is [20, N, 32] and s is [20, N, k]
        x, s = x * mask, s * mask # x*mask = [20, N, 32]*[20, N, 1] = [20, N, 32] s*mask = [20, N, k]*[20, N, 1] = [20, N, k]
        #print("x and s sizes after multiplying by mask", x.size(), s.size()

    # CT regularization
    # Calculate degree d_flat and degree matrix d
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat)+EPS  # d torch.Size([20, N, N]) 
    #print("d size", d.size())

    # Calculate Laplacian L = D - A 
    L = d - adj
    #print("Laplacian", L[1,:,:])

    # Calculate out_adj as A_CT = S.T*L*S
    # out_adj: this tensor contains A_CT = S.T*L*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), L), s) #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal 

    # Calculate CT_num 
    CT_num = _rank3_trace(out_adj) # mincut_num torch.Size([20]) one sum over each graph
    #print("CT_num size", CT_num.size())
    #print("CT_num", CT_num)
    # Calculate CT_den 
    CT_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d ), s))+EPS # [20, k, N]*[20, N, N]->[20, k, N]*[20, N, k] -> [20] one sum over each graph
    #print("CT_den size", CT_den.size())
    #print("CT_den", CT_den)

    # Calculate CT_dist (distance matrix)
    CT_dist = torch.cdist(s,s) # [20, N, k], [20, N, k]-> [20,N,N]
    #print("CT_dist",CT_dist)

    # Calculate Vol (volumes): one per graph 
    vol = _rank3_trace(d) # torch.Size([20]) 
    #print("vol size", vol.size())


    #print("vol_flat size", vol_flat.size())
    vol = _rank3_trace(d)  # d torch.Size([20, N, N]) 
    #print("vol size", vol.size())
    #print("vol", vol)

    # Calculate out_adj as CT_dist*(N-1)/vol(G)
    N = adj.size(1)
    #CT_dist = (CT_dist*(N-1)) / vol.unsqueeze(1).unsqueeze(1)
    CT_dist = (CT_dist) / vol.unsqueeze(1).unsqueeze(1)
    #CT_dist = (CT_dist) / ((N-1)*vol).unsqueeze(1).unsqueeze(1)

    #print("R_dist",CT_dist)

    # Mask with adjacency if proceeds 
    adj = CT_dist*adj
    #adj = CT_dist
    
    # Losses
    CT_loss = CT_num / CT_den
    CT_loss = torch.mean(CT_loss) # Mean over 20 graphs!
    #print("CT_loss", CT_loss)
    
    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  #[20, k, N]*[20, N, k]-> [20, k, k]
    #print("ss size", ss.size())
    i_s = torch.eye(k).type_as(ss) # [k, k]
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s )  
    #print("ortho_loss size", ortho_loss.size()) # [20] one sum over each graph
    ortho_loss = torch.mean(ortho_loss)
    #print("ortho_loss", ortho_loss) 
    
    return adj, CT_loss, ortho_loss # [20, k, 32], [20, B, N], [1], [1]
