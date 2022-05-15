import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from CT_layer import dense_CT_rewiring
from MinCut_Layer import dense_mincut_pool



@torch.no_grad()
def test_readout_embedd(modelo, loader, device):
    test_predictions = []
    test_labels = []
    modelo.eval()
    correct = 0
    for i,data in enumerate(loader):
        data = data.to(device)
        pred, mc_loss, o_loss = modelo(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + mc_loss + o_loss
        
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        test_predictions.extend(pred.max(dim=1)[1].tolist())
        test_labels.extend(data.y.detach().cpu().numpy())
        
        if i == 0:
            test_embeddings = modelo.readout
        else:
            test_embeddings = torch.cat((test_embeddings,modelo.readout.detach()), 0)
        
        #print(modelo.emb.shape)
    return loss.detach().cpu(), correct / len(loader.dataset), test_embeddings, test_labels, test_predictions


def print_readout_embeddings(train_embedd, test_embedd, train_labels, test_labels, train_pred, test_pred, num_categories, save_path):
    
    train_embedd = np.array(train_embedd)
    train_labels = np.array(train_labels)
    test_embedd = np.array(test_embedd)
    test_labels = np.array(test_labels)

    tsne = TSNE(n_components=2, verbose=0, perplexity=80, learning_rate=800)
    tsne_results = tsne.fit_transform(np.vstack([train_embedd,test_embedd]))
    
    train_embedd_2dim = tsne_results[:len(train_embedd)]
    test_embedd_2dim = tsne_results[len(train_embedd):]

    cmap = cm.get_cmap('Set2')
    fig, ax = plt.subplots(1, 2, figsize=(17,8))

    for lab in range(num_categories):
        train_indices_lab = train_labels == lab
        test_indices_lab = test_labels == lab
        
        ax[0].scatter(train_embedd_2dim[train_indices_lab,0], train_embedd_2dim[train_indices_lab,1],
                   c=np.array(cmap(lab)).reshape(1,4), label = f"{lab}_train" ,alpha=0.5, marker ='o')
        #ax[0].scatter(test_embedd_2dim[test_indices_lab,0], test_embedd_2dim[test_indices_lab,1],
        #           c=np.array(cmap(lab)).reshape(1,4), label = f"{lab}_test" ,alpha=0.5, marker ='x')
        ax[0].set_title('Real Label')
        
    for lab in range(num_categories):
        train_indices_pred = train_pred == lab
        test_indices_pred = test_pred == lab
        
        acc_test = (np.sum(test_pred == test_labels)/len(test_labels))*100
        acc_train = (np.sum(train_pred == train_labels)/len(train_pred))*100
        
        #ax[1].scatter(train_embedd_2dim[train_indices_pred,0], train_embedd_2dim[train_indices_pred,1],
        #           c=np.array(cmap(lab)).reshape(1,4), label = f"{lab}_train" ,alpha=0.5, marker ='o')
        ax[1].scatter(test_embedd_2dim[test_indices_pred,0], test_embedd_2dim[test_indices_pred,1],
                   c=np.array(cmap(lab)).reshape(1,4), label = f"{lab}_test" ,alpha=0.5, marker ='x')
        ax[1].set_title(f'Predicted\n$Acc_{{train}}={acc_train:.1f}$ - $Acc_{{test}}={acc_test:.1f}$')

        ax[1].legend(fontsize='large', markerscale=2)
    plt.savefig(f"""{save_path}.jpg""")


class CTNet_readout_embedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k_centers, hidden_channels=32):
        super(CTNet_readout_embedding, self).__init__()
    
        self.lin1 = Linear(in_channels, hidden_channels)
        num_of_centers1 =  k_centers # k1 #order of number of nodes
        self.pool1 = Linear(hidden_channels, num_of_centers1)
        #self.CT = CTLayer()
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        num_of_centers2 =  16 # k2 #mincut 
        self.pool2 = Linear(hidden_channels, num_of_centers2)
        #self.MinCut = MinCutLayer()
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels) # MLPs towards out 
        self.lin3 = Linear(hidden_channels, out_channels)
        
        self.readout = torch.zeros(0)#Creamos la variable que recoge nuestros embedings

    def forward(self, x, edge_index, batch):    # x torch.Size([N, N]),  data.batch  torch.Size([661])  
        # Make all adjacencies of size NxN 
        adj = to_dense_adj(edge_index, batch) # adj torch.Size(B, N, N])
        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40:
        x, mask = to_dense_batch(x, batch) # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20

        x = self.lin1(x)
        # First mincut pool for computing Fiedler adn rewire 
        s1  = self.pool1(x)

        if torch.isnan(adj).any():
            print("adj nan")
        if torch.isnan(x).any():
            print("x nan")
        
        # CT REWIRING
        adj, CT_loss, ortho_loss1 = dense_CT_rewiring(x, adj, s1, mask) # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])

        # CONV1: Now on x and rewired adj: 
        x = self.conv1(x, adj) #out: x torch.Size([20, N, F'=32])

        # MLP of k=16 outputs s
        s2 = self.pool2(x) # s torch.Size([20, N, k])
        
        # MINCUT_POOL
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask) # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])

        # CONV2: Now on coarsened x and adj: 
        x = self.conv2(x, adj) #out x torch.Size([20, 16, 32])
        
        # Readout for each of the 20 graphs
        x = x.sum(dim=1) # x torch.Size([20, 32])
        
        #Queremos esta x, por lo que nos la guardamos
        self.readout = x.clone()
        
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x)) # x torch.Size([20, 32])
        x = self.lin3(x) #x torch.Size([20, 2])
        #print(x.shape)
        
        CT_loss = CT_loss + ortho_loss1
        mincut_loss = mincut_loss2 + ortho_loss2
        return F.log_softmax(x, dim=-1), CT_loss, mincut_loss