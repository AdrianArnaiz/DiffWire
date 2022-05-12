import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, DenseGraphConv
from ein_utils import _rank3_diag, _rank3_trace
EPS = 1e-15

def dense_mincut_pool(x, adj, s, mask=None): # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20
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
    s = torch.softmax(s, dim=-1) # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())

    if mask is not None: # NOT None for now
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        #print("mask size", mask.size()) # [20, N, 1]
        # Mask pointwise product. Since x is [20, N, 32] and s is [20, N, k]
        x, s = x * mask, s * mask # x*mask = [20, N, 32]*[20, N, 1] = [20, N, 32] s*mask = [20, N, k]*[20, N, 1] = [20, N, k]
        #print("x and s sizes after multiplying by mask", x.size(), s.size())

    # out: this tensor contains Xpool=S.T*X (Eq. 7)
    out = torch.matmul(s.transpose(1, 2), x)  # [20, k, N] * [20, N, 32] will yield [20, k, 32]
    #print("out size", out.size())
    # out_adj: this tensor contains Apool = S.T*A*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s) #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal 

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj) # mincut_num torch.Size([20]) one sum over each graph
    #print("mincut_num size", mincut_num.size())
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat) # d torch.Size([20, N, N]) 
    #print("d size", d.size())
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) # [20, k, N]*[20, N, N]->[20, k, N]*[20, N, k] -> [20] one sum over each graph
    #print("mincut_den size", mincut_den.size())
    
    mincut_loss = -(mincut_num / mincut_den)
    #print("mincut_loss", mincut_loss)
    mincut_loss = torch.mean(mincut_loss) # Mean over 20 graphs!
    
    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  #[20, k, N]*[20, N, k]-> [20, k, k]
    #print("ss size", ss.size())
    i_s = torch.eye(k).type_as(ss) # [k, k]
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))  
    #print("ortho_loss size", ortho_loss.size()) # [20] one sum over each graph
    ortho_loss = torch.mean(ortho_loss) 
    
    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device) # range e.g. from 0 to 15 (k=16)
    # out_adj is [20, k, k]
    out_adj[:, ind, ind] = 0 # [20, k, k]  the diagnonal will be 0 now: Ahat = Apool - I_k*diag(Apool) (Eq. 8)
    #print("out_adj", out_adj[0,])

    # Final degree matrix and normalization of out_adj: Ahatpool = Dhat^{-1/2}AhatD^{-1/2} (Eq. 8)
    d = torch.einsum('ijk->ij', out_adj) #d torch.Size([20, k])
    #print("d size", d.size())
    d = torch.sqrt(d+EPS)[:, None] + EPS # d torch.Size([20, 1, k])
    #print("sqrt(d) size", d.size())
    #print( (out_adj / d).shape)  # out_adj is [20, k, k] and d is [20, 1, k] -> torch.Size([20, k, k]) 
    out_adj = (out_adj / d) / d.transpose(1, 2) # ([20, k, k] / [20, k, 1] ) -> [20, k, k]
    # out_adj torch.Size([20, k, k]) 
    #print("out_adj size", out_adj.size())
    return out, out_adj, mincut_loss, ortho_loss # [20, k, 32], [20, k, k], [1], [1]