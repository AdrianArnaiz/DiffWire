import time
import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, DenseGraphConv
from layers.utils.ein_utils import _rank3_diag, _rank3_trace
from layers.utils.approximate_fiedler import approximate_Fiedler 
from layers.utils.approximate_fiedler import NLderivative_of_lambda2_wrt_adjacency, NLfiedler_values 
from layers.utils.approximate_fiedler import derivative_of_lambda2_wrt_adjacency, fiedler_values
from layers.utils.approximate_fiedler import NLderivative_of_lambda2_wrt_adjacencyV2, NLfiedler_valuesV2

def dense_mincut_rewiring(x, adj, s, mask=None, derivative = None, EPS=1e-15, device=None): # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20
    
    k = 2 #We want bipartition to compute spectral gap
    # adj torch.Size([20, N, N]) N=Mmax
    #print("Input adj size to mincut pool", adj.size())
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj # adj torch.Size([20, N, N]) N=Mmax
    #print("Unsqueezed adj size to mincut pool", adj.size(), adj.dim()) # adj.dim() is usually 3

    # s torch.Size([20, N, k])
    s = s.unsqueeze(0) if s.dim() == 2 else s #s torch.Size([20, N, k])
    #print("Unsqueezed s size", s.size())

    s = torch.softmax(s, dim=-1) # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())
    #print("s softmax", s[0,1,:], torch.argmax(s,dim=(2)).size())
  
    # Put here the calculus of the degree matrix to optimize the complex derivative
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat) # d torch.Size([20, N, N]) 
    #print("d size", d.size())

    # Batched Laplacian 
    L = d - adj
    
    # REWIRING: UPDATING adj wrt s using derivatives -------------------------------------------------
    # Approximating the Fiedler vectors from s (assuming k=2)
    fiedlers = approximate_Fiedler(s, device)
    #print("fiedlers size", fiedlers.size())
    #print("fiedlers ", fiedlers)

    # Recalculate
    if derivative == "laplacian":
        der = derivative_of_lambda2_wrt_adjacency(fiedlers, device)
        fvalues = fiedler_values(adj, fiedlers, EPS, device)
    elif derivative == "normalized":
        #start = time.time()
        der = NLderivative_of_lambda2_wrt_adjacency(adj, d_flat, fiedlers, EPS, device)   
        fvalues = NLfiedler_values(L, d_flat, fiedlers, EPS, device)
        #print('\t\t NLderivative_of_lambda2_wrt_adjacency: {:.6f}s'.format(time.time()- start))
    elif derivative == "normalizedv2":
        der = NLderivative_of_lambda2_wrt_adjacencyV2(adj, d_flat, fiedlers, EPS, device)   
        fvalues = NLfiedler_valuesV2(L, d, fiedlers, EPS, device)

    mu = 0.01
    lambdaReg = 0.1 
    lambdaReg = 1.0 
    lambdaReg = 1.5 
    lambdaReg = 2.0
    lambdaReg = 2.5
    lambdaReg = 5.0
    lambdaReg = 3.0 
    lambdaReg = 1.0
    lambdaReg = 2.0 
    #lambdaReg = 20.0      
    
    Ac = adj.clone()
    for _ in range(5):
      #fvalues = fiedler_values(Ac, fiedlers)
      #print("Ac size", Ac.size())
      partialJ = 2*(Ac-adj) + 2*lambdaReg*der*fvalues.unsqueeze(1).unsqueeze(2) # favalues is [B], partialJ is [B, N, N]
      #print("partialJ size", partialJ.size())
      #print("diag size", torch.diag_embed(torch.diagonal(partialJ,dim1=1,dim2=2)).size())
      dJ = partialJ + torch.transpose(partialJ,1,2) - torch.diag_embed(torch.diagonal(partialJ,dim1=1,dim2=2))
      # Update adjacency
      Ac  = Ac - mu*dJ
      # Clipping: negatives to 0, positives to 1 
      #print("Ac is", Ac, Ac.size())
      #Ac = torch.clamp(Ac, min=0.0, max=1.0)
      
      #print("Esta es la antigua adj",adj)
      #print("Esta es la antigua Ac",Ac)
      
      #print("Despues mask Ac",Ac)
      #print("Despues mask Adj",adj)
      #print("Mayores que 0",(Ac>0).sum()) #20,16,40
      #print("Menores que 0",(Ac<=0).sum()) 
      Ac = torch.softmax(Ac, dim=-1)
      Ac = Ac*adj
    #print("Min Fiedlers",min(fvalues))
    #print("NUeva salida",Ac)
    
    # out_adj: this tensor contains Apool = S.T*A*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s) #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal 

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj) # mincut_num torch.Size([20]) one sum over each graph
    #print("mincut_num size", mincut_num.size())
    #d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    #print("d_flat size", d_flat.size())
    #d = _rank3_diag(d_flat) # d torch.Size([20, N, N]) 
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
    
    """# Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device) # range e.g. from 0 to 15 (k=16)
    # out_adj is [20, k, k]
    out_adj[:, ind, ind] = 0 # [20, k, k]  the diagnonal will be 0 now: Ahat = Apool - I_k*diag(Apool) (Eq. 8)
    #print("out_adj", out_adj[0,])

    # Final degree matrix and normalization of out_adj: Ahatpool = Dhat^{-1/2}AhatD^{-1/2} (Eq. 8)
    d = torch.einsum('ijk->ij', out_adj) #d torch.Size([20, k])
    #print("d size", d.size())
    d = torch.sqrt(d)[:, None] + EP S # d torch.Size([20, 1, k])
    #print("sqrt(d) size", d.size())
    #print( (out_adj / d).shape)  # out_adj is [20, k, k] and d is [20, 1, k] -> torch.Size([20, k, k]) 
    out_adj = (out_adj / d) / d.transpose(1, 2) # ([20, k, k] / [20, k, 1] ) -> [20, k, k]
    # out_adj torch.Size([20, k, k]) 
    #print("out_adj size", out_adj.size())"""
    return  Ac, mincut_loss, ortho_loss # [20, k, 32], [20, k, k], [1], [1]
    #return out, out_adj, mincut_loss, ortho_loss # [20, k, 32], [20, k, k], [1], [1]