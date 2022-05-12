from math import ceil
from nets import CTNet, GAPNet
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import time

######################################################
train_log_file = "ComplexDerivativeTEST_"
RandList = [12345, 42345, 64345, 54345, 74345, 47345, 54321, 14321, 94321, 84328]
#RandList = RandList[:]

DERIVATIVE = "laplacian" #laplacian or normalized

BATCH_SIZE = 100
DATASET = 'CIFAR10'
DATA_LIMIT = 50000 if DATASET=='MNIST' else 40000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:1"

train_log_file = train_log_file + DATASET +time.strftime('%d_%m_%y__%H_%M') + '.txt'
######################################################

def train(epoch, loader):
    model.train()
    loss_all = 0
    correct = 0
    #i = 0
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        out, mc_loss, o_loss = model(data.x, data.edge_index, data.batch) # data.batch  torch.Size([783])

        loss = F.nll_loss(out, data.y.view(-1)) + mc_loss + o_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()

        correct += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item() #accuracy in train AFTER EACH BACH
    #print("Training graphs per epoch", nG)
    return loss_all / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        #print(data.x)
        pred, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        #print(next(model.parameters()).device)
        #print(data.x.device)
        loss = F.nll_loss(pred, data.y.view(-1)) + mc_loss + o_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)


######################################################
print(device)

dataset = GNNBenchmarkDataset(root='data/GNNBenchmarkDataset', name=DATASET) #MNISTo CIFAR10
data = dataset[0]  # Get the first graph object.
dataset = dataset.shuffle()

train_dataset = dataset[:DATA_LIMIT] #MNIST : 50000 - CIFAR: 40000
test_dataset = dataset[DATA_LIMIT:] 

#torch.autograd.set_detect_anomaly(True)
ExperimentResult = []

f = open(train_log_file, 'w') #clear file
f.close()
for e in range(len(RandList)):
    model = GAPNet(dataset.num_features, dataset.num_classes, derivative=DERIVATIVE, device=device).to(device)
    #model = CTNet(dataset.num_features, dataset.num_classes).to(device)
    #model = GAPNet(dataset.new_num_features, dataset.new_num_classes).to(device)
    #model = NetCT(dataset.new_num_features, dataset.new_num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)#
    train_loader = DataLoader(train_dataset.shuffle(), batch_size=BATCH_SIZE, shuffle=True) # Original 64
    test_loader = DataLoader(test_dataset.shuffle(), batch_size=BATCH_SIZE, shuffle=False)  # Original 64
    #train_dataset = train_dataset.shuffle()
    #test_dataset = test_dataset.shuffle()
    optimizer.zero_grad()
    torch.manual_seed(RandList[e])
    print("Experimen run", RandList[e])

    for epoch in range(1, 60):
        start_time_epoch = time.time()
        train_loss, train_acc = train(epoch, train_loader) # return also train_acc_t if want Accuracy after BATCH
        #_, train_acc = test(train_loader)
        test_loss, test_acc = test(test_loader)
        time_lapse = time.time() - start_time_epoch

        f = open(train_log_file, 'a')
        print('Epoch: {:03d}, '
                'Train Loss: {:.3f}, Train Acc: {:.3f}, '
                'Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, train_loss,
                                                            train_acc, test_loss,
                                                            test_acc), file=f)
        print('Epoch: {:03d}, '
                'Train Loss: {:.3f}, Train Acc: {:.3f}, '
                'Test Loss: {:.3f}, Test Acc: {:.3f}, Time: {:.2f}'.format(epoch, train_loss,
                                                            train_acc, test_loss,
                                                            test_acc, time_lapse))
        f.close()
        
    ExperimentResult.append(test_acc)
    f = open(train_log_file, 'a')
    print('Result of run {:.3f} is {:.3f}'.format(e,test_acc), file=f)
    f.close()

f = open(train_log_file, 'a')
print('Test Acc of 10 execs {}'.format(ExperimentResult), file=f)
f.close()
