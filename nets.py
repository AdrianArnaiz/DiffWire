import time
from GAP_layer import dense_mincut_rewiring
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from CT_layer import dense_CT_rewiring
from MinCut_Layer import dense_mincut_pool

class GAPNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, derivative=None, EPS=1e-15, device=None):
        super(GAPNet, self).__init__()
        self.device = device
        self.derivative = derivative
        self.EPS = EPS
        # GCN Layer - MLP - Dense GCN Layer
        #self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_of_centers2 =  16 # k2
        #num_of_centers2 =  10 # k2
        #num_of_centers2 =  5 # k2
        num_of_centers1 =  2 # k1 #Fiedler vector
        # The degree of the node belonging to any of the centers
        self.pool1 = Linear(hidden_channels, num_of_centers1) 
        self.pool2 = Linear(hidden_channels, num_of_centers2) 
        # MLPs towards out 
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

        # Input: Batch of 20 graphs, each node F=3 features 
        #        N1 + N2 + ... + N2 = 661
        # TSNE here?
    def forward(self, x, edge_index, batch):    # x torch.Size([N, N]),  data.batch  torch.Size([661])  

        # Make all adjacencies of size NxN 
        adj = to_dense_adj(edge_index, batch)   # adj torch.Size(B, N, N])
        #print("adj_size", adj.size())
        #print("adj",adj)
        

        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40: 
        #print("x size", x.size())
        x, mask = to_dense_batch(x, batch)      # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20
        #print("x size", x.size())

        x = self.lin1(x)
        # First mincut pool for computing Fiedler adn rewire 
        s1  = self.pool1(x)
        #s1 = torch.variable()#s1 torch.Size([20, N, k1=2)
        #s1 = Variable(torch.randn(D_in, H).type(float16), requires_grad=True)
        #print("s 1st pool",s1)
        #print("s 1st pool size", s1.size())

        if torch.isnan(adj).any():
          print("adj nan")
        if torch.isnan(x).any():
          print("x nan") 

        
        # REWIRING
        #start = time.time()
        adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask, derivative = self.derivative, EPS=self.EPS, device=self.device) # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])
        #print('\t\tdense_mincut_rewiring: {:.6f}s'.format(time.time()- start))
        #print("x",x)
        #print("adj",adj)
        #print("x and adj sizes", x.size(), adj.size())
        #adj = torch.softmax(adj, dim=-1)
        #print("adj softmaxed", adj)

        # CONV1: Now on x and rewired adj: 
        x = self.conv1(x, adj) #out: x torch.Size([20, N, F'=32])
        #print("x_1 ", x)
        #print("x_1 size", x.size())
        
        # MLP of k=16 outputs s
        #print("adj_size", adj.size())
        s2 = self.pool2(x) # s torch.Size([20, N, k])
        #print("s 2nd pool", s2)
        #print("s 2nd pool size", s2.size())
        #adj = torch.softmax(adj, dim=-1)
        
        
        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        #x, adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask) # x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS) # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        #print("lossses2",mincut_loss2, ortho_loss2)
        #print("mincut pool x", x)
        #print("mincut pool adj", adj)
        #print("mincut pool x size", x.size())
        #print("mincut pool adj size", adj.size()) # Some nan in adjacency: maybe comming from the rewiring-> dissapear after clipping
        

        # CONV2: Now on coarsened x and adj: 
        x = self.conv2(x, adj) #out x torch.Size([20, 16, 32])
        #print("x_2", x)
        #print("x_2 size", x.size())
        
        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1) # x torch.Size([20, 32])
        #print("mean x_2 size", x.size())
        
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x)) # x torch.Size([20, 32])
        #print("final x1 size", x.size())
        x = self.lin3(x) #x torch.Size([20, 2])
        #print("final x2 size", x.size())
        #print("losses: ", mincut_loss1, mincut_loss2, ortho_loss2, mincut_loss2)
        mincut_loss = mincut_loss1 + mincut_loss2
        ortho_loss = ortho_loss1 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), mincut_loss, ortho_loss


class CTNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k_centers, hidden_channels=32, EPS=1e-15):
        super(CTNet, self).__init__()
        self.EPS=EPS
        # GCN Layer - MLP - Dense GCN Layer
        #self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        
        # The degree of the node belonging to any of the centers
        num_of_centers1 =  k_centers # k1 #order of number of nodes
        self.pool1 = Linear(hidden_channels, num_of_centers1)
        num_of_centers2 =  16 # k2 #mincut 
        self.pool2 = Linear(hidden_channels, num_of_centers2) 

        # MLPs towards out 
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
 

    def forward(self, x, edge_index, batch):    # x torch.Size([N, N]),  data.batch  torch.Size([661])  
        # Make all adjacencies of size NxN 
        adj = to_dense_adj(edge_index, batch)   # adj torch.Size(B, N, N])
        #print("adj_size", adj.size())
        #print("adj",adj)

        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40: 
        #print("x size", x.size())
        x, mask = to_dense_batch(x, batch)      # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20
        #print("x size", x.size())

        x = self.lin1(x)
        # First mincut pool for computing Fiedler adn rewire 
        s1  = self.pool1(x)
        #s1 = torch.variable()#s1 torch.Size([20, N, k1=2)
        #s1 = Variable(torch.randn(D_in, H).type(float16), requires_grad=True)
        #print("s 1st pool",s1)
        #print("s 1st pool size", s1.size())

        if torch.isnan(adj).any():
          print("adj nan")
        if torch.isnan(x).any():
          print("x nan")
        
        # CT REWIRING
        adj, CT_loss, ortho_loss1 = dense_CT_rewiring(x, adj, s1, mask, EPS = self.EPS) # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])
        
        #print("CT_loss, ortho_loss1", CT_loss, ortho_loss1)
        #print("x",x)
        #print("adj",adj)
        #print("x and adj sizes", x.size(), adj.size())
        #adj = torch.softmax(adj, dim=-1)
        #print("adj softmaxed", adj)

        # CONV1: Now on x and rewired adj: 
        x = self.conv1(x, adj) #out: x torch.Size([20, N, F'=32])
        #print("x_1 ", x)
        #print("x_1 size", x.size())
        
        # MLP of k=16 outputs s
        #print("adj_size", adj.size())
        s2 = self.pool2(x) # s torch.Size([20, N, k])
        #print("s 2nd pool", s2)
        #print("s 2nd pool size", s2.size())
        #adj = torch.softmax(adj, dim=-1)
        
        
        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        #x, adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask) # x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS) # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        #print("lossses2",mincut_loss2, ortho_loss2)
        #print("mincut pool x", x)
        #print("mincut pool adj", adj)
        #print("mincut pool x size", x.size())
        #print("mincut pool adj size", adj.size()) # Some nan in adjacency: maybe comming from the rewiring-> dissapear after clipping
        

        # CONV2: Now on coarsened x and adj: 
        x = self.conv2(x, adj) #out x torch.Size([20, 16, 32])
        #print("x_2", x)
        #print("x_2 size", x.size())
        
        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1) # x torch.Size([20, 32])
        #print("mean x_2 size", x.size())
        
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x)) # x torch.Size([20, 32])
        #print("final x1 size", x.size())
        x = self.lin3(x) #x torch.Size([20, 2])
        #print("final x2 size", x.size())
        CT_loss = CT_loss + ortho_loss1
        mincut_loss = mincut_loss2 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), CT_loss, mincut_loss


class MinCutNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, EPS=1e-15):
        super(MinCutNet, self).__init__()
        self.EPS=EPS
        # GCN Layer - MLP - Dense GCN Layer
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        
        # The degree of the node belonging to any of the centers
        num_of_centers2 =  16 # k2 #mincut 
        self.pool2 = Linear(hidden_channels, num_of_centers2) 

        # MLPs towards out 
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
 

    def forward(self, x, edge_index, batch):    # x torch.Size([N, N]),  data.batch  torch.Size([661])  
    
        # Make all adjacencies of size NxN 
        adj = to_dense_adj(edge_index, batch)   # adj torch.Size(B, N, N])
        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40: 
        x, mask = to_dense_batch(x, batch)      # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20

        x = self.lin1(x)

        if torch.isnan(adj).any():
          print("adj nan")
        if torch.isnan(x).any():
          print("x nan")

        # CONV1: Now on x and rewired adj: 
        x = self.conv1(x, adj) #out: x torch.Size([20, N, F'=32])
        
        # MLP of k=16 outputs s
        s2 = self.pool2(x) # s torch.Size([20, N, k])

        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS) # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        
        # CONV2: Now on coarsened x and adj: 
        x = self.conv2(x, adj) #out x torch.Size([20, 16, 32])
        
        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1) # x torch.Size([20, 32])
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x)) # x torch.Size([20, 32])
        x = self.lin3(x) #x torch.Size([20, 2])
        
        mincut_loss = mincut_loss2 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), mincut_loss2, ortho_loss2