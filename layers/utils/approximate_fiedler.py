import torch
import numpy as np
import time

from layers.utils.ein_utils import _rank3_diag

def approximate_Fiedler(s, device=None): # torch.Size([20, N, k]) One k for each N of each graph (asume k=2)
  """
  Calculate approximate fiedler vector from S matrix. S in R^{B x N x 2} and fiedler vector S in R^{B x N}
  """
  s_0 = s.size(0) #number of graphs
  s_1 = s.size(1)
  maxcluster = torch.argmax(s,dim=(2)) # torch.Size([20, N]) with binary values {0,1} if k=2
  trimmed_s = torch.FloatTensor(s_0,s_1).to(device)
  #print('\t'*4,'[DEVICES] s device', s.device,' -- trimmed_s device', trimmed_s.device,' -- maxcluster device', trimmed_s.device)
  trimmed_s[maxcluster==1] = -1/np.sqrt(float(s_1))
  trimmed_s[maxcluster==0] = 1/np.sqrt(float(s_1))  
  return trimmed_s

def NLderivative_of_lambda2_wrt_adjacency(adj, d_flat, fiedlers, EPS, device): # fiedlers torch.Size([20, N])
  """
  Complex derivative

  Args:
      adj (_type_): _description_
      d_flat (_type_): _description_
      fiedlers (_type_): _description_

  Returns:
      _type_: _description_
  """
  N = fiedlers.size(1)
  B = fiedlers.size(0)
  # Batched structures for the complex derivative  
  d_flat2 = torch.sqrt(d_flat+EPS)[:, None] + EPS # d torch.Size([B, 1, N])
  #print("first d_flat2 size", d_flat2.size())
  Ahat = (adj/d_flat2.transpose(1, 2)) # [B, N, N] / [B, N, 1] -> [B, N, N]
  AhatT = (adj.transpose(1,2)/d_flat2.transpose(1, 2)) # [B, N, N] / [B, N, 1] -> [B, N, N]
  dinv = 1/(d_flat + EPS)[:, None]
  dder = -0.5*dinv*d_flat2
  dder = dder.transpose(1,2) # [B, N, 1]
  # Storage 
  derivatives = torch.FloatTensor(B, N, N).to(device)
  
  for b in range(B):
    # Eigenvectors
    u2 = fiedlers[b,:]
    u2 = u2.unsqueeze(1) # [N, 1]
    #u2 = u2.to(device) #its already in device because fiedlers is already in device
    #print("size of u2", u2.size())

    # First term central: [N,1]x ([1,N]x[N,N]x[N,1]) x [N,1]
    firstT = torch.matmul(torch.matmul(u2.T, AhatT[b,:,:]), u2) # [1,N]x[N,N]x[N,1] -> [1]
    #print("first term central size", firstT.size())
    firstT = torch.matmul(torch.matmul(dder[b,:], firstT), torch.ones(N).unsqueeze(0).to(device))

    # Second term
    secT = torch.matmul(torch.matmul(u2.T, Ahat[b,:,:]), u2) # [1,N]x[N,N]x[N,1] -> [1]
    #print("second term central size", secT.size())
    secT = torch.matmul(torch.matmul(dder[b,:], secT), torch.ones(N).unsqueeze(0).to(device))

    # Third term
    u2u2T = torch.matmul(u2,u2.T) # [N,1] x [1,N] -> [N,N]
    #print("u2u2T size", u2u2T.size())
    #print("d_flat2[b,:] size", d_flat2[b,:].size())
    Du2u2TD = (u2u2T / d_flat2[b,:]) / d_flat2[b,:].transpose(0, 1)
    #print("size of Du2u2TD", Du2u2TD.size())
    # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL 
    #dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)),torch.ones(N,N)) - u2u2T
    dl2 = firstT + secT + Du2u2TD
    # Symmetrize and subtract the diag since it is an undirected graph
    #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
    derivatives[b,:,:] = -dl2
  return derivatives # derivatives torch.Size([20, N, N])

def NLfiedler_values(L, d_flat, fiedlers, EPS, device): # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
  N = fiedlers.size(1)
  B = fiedlers.size(0)
  #print("original fiedlers size", fiedlers.size())
  
  # Batched Fiedlers 
  d_flat2 = torch.sqrt(d_flat+EPS)[:, None] + EPS # d torch.Size([B, 1, N])
  #print("d_flat2 size", d_flat2.size())
  fiedlers = fiedlers.unsqueeze(2) # [B, N, 1]
  #print("fiedlers size", fiedlers.size())
  fiedlers_hats = (fiedlers/d_flat2.transpose(1, 2))  # [B, N, 1] / [B, N, 1] -> [B, N, 1]
  gfiedlers_hats = (fiedlers*d_flat2.transpose(1, 2))  # [B, N, 1] * [B, N, 1] -> [B, N, 1]
  #print("fiedlers size", fiedlers_hats.size())
  #print("gfiedlers size", gfiedlers_hats.size())
  
  #Laplacians = torch.FloatTensor(B, N, N)
  fiedler_values = torch.FloatTensor(B).to(device)
  for b in range(B):
    f = fiedlers_hats[b,:]
    g = gfiedlers_hats[b,:]
    num = torch.matmul(f.T,torch.matmul(L[b,:,:],f)) # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
    den = torch.matmul(g.T,g)
    #print("num fied", num.size())
    #print("den fied", den.size())
    #print("g size", g.size())
    #print("f size", f.size())
    #print("L size", L[b,:,:].size())
    fiedler_values[b] = N*torch.abs(num/(den + EPS))
  return fiedler_values # torch.Size([B])
  
def derivative_of_lambda2_wrt_adjacency(fiedlers, device): # fiedlers torch.Size([20, N])
  """
  Simple derivative
  """
  N = fiedlers.size(1)
  B = fiedlers.size(0)
  derivatives = torch.FloatTensor(B, N, N).to(device)
  for b in range(B):
    u2 = fiedlers[b,:]
    u2 = u2.unsqueeze(1)
    #print("size of u2", u2.size())
    u2u2T = torch.matmul(u2,u2.T)
    #print("size of u2u2T", u2u2T.size())
    # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL 
    dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)),torch.ones(N,N).to(device)) - u2u2T
    # Symmetrize and subtract the diag since it is an undirected graph
    #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
    derivatives[b,:,:] = dl2
    
  return derivatives # derivatives torch.Size([20, N, N])

def fiedler_values(adj, fiedlers, EPS, device): # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
  N = fiedlers.size(1)
  B = fiedlers.size(0)
  #Laplacians = torch.FloatTensor(B, N, N)
  fiedler_values = torch.FloatTensor(B).to(device)
  for b in range(B):
    # Compute un-normalized Laplacian
    A = adj[b,:,:]
    D = A.sum(dim=1)
    D = torch.diag(D)
    L = D - A
    #Laplacians[b,:,:] = L
    #if torch.min(A)<0:
    #  print("Negative adj")
    # Compute numerator 
    f = fiedlers[b,:].unsqueeze(1)
    #f = f.to(device)
    num = torch.matmul(f.T,torch.matmul(L,f)) # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
    # Create complete graph Laplacian
    CA = torch.ones(N,N).to(device)-torch.eye(N).to(device)
    CD = CA.sum(dim=1)
    CD = torch.diag(CD)
    CL = CD - CA
    CL = CL.to(device)
    # Compute denominator 
    den = torch.matmul(f.T,torch.matmul(CL,f))
    fiedler_values[b] = N*torch.abs(num/(den + EPS))

  return fiedler_values # torch.Size([B])


def NLderivative_of_lambda2_wrt_adjacencyV2(adj, d_flat, fiedlers, EPS, device): # fiedlers torch.Size([20, N])
    """
    Complex derivative
    Args:
      adj (_type_): _description_
      d_flat (_type_): _description_
      fiedlers (_type_): _description_
    Returns:
      _type_: _description_
    """
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    # Batched structures for the complex derivative
    d_flat2 = torch.sqrt(d_flat+EPS)[:, None] + EPS # d torch.Size([B, 1, N])
    d_flat = d_flat2.squeeze(1)
    #print("first d_flat2 size", d_flat2.size())
    d_half =  _rank3_diag(d_flat) # d torch.Size([B, N, N])
    #print("d size", d.size())
    Ahat = (adj/d_flat2.transpose(1, 2)) # [B, N, N] / [B, N, 1] -> [B, N, N]
    AhatT = (adj.transpose(1,2)/d_flat2.transpose(1, 2)) # [B, N, N] / [B, N, 1] -> [B, N, N]
    dinv = 1/(d_flat + EPS)[:, None]
    dder = -0.5*dinv*d_flat2
    dder = dder.transpose(1,2) # [B, N, 1]
    # Storage
    derivatives = torch.FloatTensor(B, N, N).to(device)
    for b in range(B):
      # Eigenvectors
      u2 = fiedlers[b,:]
      u2 = u2.unsqueeze(1) # [N, 1]
      #u2 = u2.to(device) #its already in device because fiedlers is already in device
      #print("size of u2", u2.size())
      # First term central: [N,1]x ([1,N]x[N,N]x[N,1]) x [N,1]
      firstT = torch.matmul(torch.matmul(u2.T, torch.matmul(d_half[b,:,:], AhatT[b,:,:])), u2) # [1,N]x[N,N]x[N,1] -> [1]
      #print("first term central size", firstT.size())
      firstT = torch.matmul(torch.matmul(dder[b,:], firstT), torch.ones(N).unsqueeze(0).to(device))
      #print("first term  size", firstT.size())
      # Second term
      secT = torch.matmul(torch.matmul(u2.T, torch.matmul(d_half[b,:,:], Ahat[b,:,:])), u2) # [1,N]x[N,N]x[N,1] -> [1]
      #print("second term central size", secT.size())
      secT = torch.matmul(torch.matmul(dder[b,:], secT), torch.ones(N).unsqueeze(0).to(device))
      # Third term
      Du2u2TD = torch.matmul(u2,u2.T) # [N,1] x [1,N] -> [N,N]
      #print("Du2u2T size", u2u2T.size())
      #print("d_flat2[b,:] size", d_flat2[b,:].size())
      #Du2u2TD = (u2u2T / d_flat2[b,:]) / d_flat2[b,:].transpose(0, 1)
      #print("size of Du2u2TD", Du2u2TD.size())
      # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL
      #dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)),torch.ones(N,N)) - u2u2T
      dl2 = firstT + secT + Du2u2TD
      # Symmetrize and subtract the diag since it is an undirected graph
      #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
      derivatives[b,:,:] = -dl2
    return derivatives # derivatives torch.Size([20, N, N])

    
def NLfiedler_valuesV2(L, d, fiedlers, EPS, device): # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
  N = fiedlers.size(1)
  B = fiedlers.size(0)
  #print("original fiedlers size", fiedlers.size())
  #print("d size", d.size())
  #Laplacians = torch.FloatTensor(B, N, N)
  fiedler_values = torch.FloatTensor(B).to(device)
  for b in range(B):
    f = fiedlers[b,:].unsqueeze(1)
    num = torch.matmul(f.T,torch.matmul(L[b,:,:],f)) # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
    den = torch.matmul(f.T,torch.matmul(d[b,:,:], f)) # d is [N, N], f is [N,1], f.T is [1,N] -> [1,N] x [N, N] x [N, 1] is [1]
    fiedler_values[b] = N*torch.abs(num/(den + EPS))
    """print(f.shape)
    print(f.T.shape)
    print(num.shape, num)
    print(den.shape, den)
    print((N*torch.abs(num/(den + EPS))).shape)
    exit()"""
  return fiedler_values # torch.Size([B])

