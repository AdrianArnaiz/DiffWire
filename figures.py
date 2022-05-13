import torch
from nets import CTNet, GAPNet
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from transform_features import FeatureDegree
from torch_geometric.datasets import TUDataset

@torch.no_grad()
def test(modelo, loader, device):
    modelo.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred, mc_loss, o_loss = modelo(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + mc_loss + o_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
    print(correct)
    return loss, correct / len(loader.dataset)

########################
dataset = TUDataset(root='data_colab/TUDataset',name="REDDIT-BINARY", pre_transform=FeatureDegree(), use_node_attr=True)
BATCH_SIZE = 64
num_of_centers = 420

"""dataset = TUDataset(root='data_colab/TUDataset',name="MUTAG")
BATCH_SIZE = 31
num_of_centers = 17"""
######################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model =  CTNet(dataset.num_features, dataset.num_classes, k_centers=num_of_centers).to(device)
model.load_state_dict(torch.load("models/REDDIT-BINARY_CTNet_iter0.pth"))
#model.load_state_dict(torch.load("models/MUTAG_CTNet_iter0.pth"))
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # Original 64
loss_test, acc_test = test(model, test_loader, device)
print('Test', loss_test, acc_test)


