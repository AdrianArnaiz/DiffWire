from math import ceil
import random
from nets import CTNet, GAPNet
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from no_features_class import * 
import time
import argparse
'''
    Dataset arguments:
        -Name:
                *TUDataset:
                            + No features: [REDDEDIT BINARY, IMBD BINARY, COLLAB]
                            + Featured: [MUTAG, ENZYMES, PROTEINS]
                *Benchmark: [MNIST,CIFAR10]
    Model arguments:
        -Name:
            MC(num_features,num_classes)
            GAPNet(new_num_features, dataset.new_num_classes,derivative)
            CTNet(num_features,num_classes)
    Other arguments:
        Lr
        weight decay

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cuda:1"
BATCH_SIZE = 0
TRAIN_SPLIT = 0
N_EPOCH = 60
RandList = [12345, 42345, 64345, 54345, 74345, 47345, 54321, 14321, 94321, 84328]
No_Features = ["COLLAB","IMDB-BINARY","REDDIT-BINARY"]
parser = argparse.ArgumentParser()
parser.add_argument(
        "--dataset",
        default="CIFAR10",
        choices=["MUTAG","ENZYMES","PROTEINS","CIFAR10","MNIST","COLLAB","IMDB-BINARY","REDDIT-BINARY"],
        help="nada",
)
parser.add_argument(
    "--model",
    default="CTNet",
    choices=["CTNet","GAPNet"],
    help="nada",
)
parser.add_argument(
    "--derivative",
    default="laplacian",
    choices=["laplacian","normalized"],
    help="nada",
)
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=1e-4, help="Outer weight decay rate of model"
    )
args = parser.parse_args()

#Procesing dataset
No_Features = ["COLLAB","IMDB-BINARY","REDDIT-BINARY"]
if args.dataset not in GNNBenchmarkDataset.names:
    if args.dataset not in No_Features:
        if args.dataset =="MUTAG":
            TRAIN_SPLIT = 150
            BATCH_SIZE = 32
            num_of_centers = 17 #mean number of nodes according to PyGeom
        if args.dataset =="ENZYMES":
            TRAIN_SPLIT = 500
            BATCH_SIZE = 32
            num_of_centers = 32 #mean number of nodes according to PyGeom
        if args.dataset =="PROTEINES":
            TRAIN_SPLIT = 1000
            BATCH_SIZE = 64
            num_of_centers = 39 #mean number of nodes according to PyGeom
    else:
        datasetGNN = TUDataset(root='data/TUDataset', name=args.dataset)
        dataset = TUDatasetFeatures(root='data/TUDataset', name=args.dataset,dataset=datasetGNN)
        
        if args.dataset =="IMDB-BINARY":
            TRAIN_SPLIT = 800
            BATCH_SIZE = 64
            num_of_centers = 20 #mean number of nodes according to PyGeom
        elif args.dataset == "REDDIT-BINARY":
            TRAIN_SPLIT = 1500
            BATCH_SIZE = 64
            num_of_centers = 420 #mean number of nodes according to PyGeom
        else:
            raise Exception("Not dataset in list of datasets")
else: #GNNBenchmarkDataset
    dataset = GNNBenchmarkDataset(root='data/GNNBenchmarkDataset', name=args.dataset) #MNISTo CIFAR10
    if args.dataset =="MNIST":
        TRAIN_SPLIT = 50000
        BATCH_SIZE =100
    elif args.dataset == "CIFAR10":
        TRAIN_SPLIT = 40000
        BATCH_SIZE = 100
    #nothing
#Mejorable, se puede hacer de una en el run10
train_dataset = dataset[:TRAIN_SPLIT] #MNIST : 50000 - CIFAR: 40000
test_dataset = dataset[TRAIN_SPLIT:] 
#Procesing model
arquitecture = globals()[args.model]
if args.model == 'CTNet':
        model = CTNet(dataset.num_features, dataset.num_classes, k_centers=num_of_centers).to(device)
elif args.model == 'GAPNet':
    model = GAPNet(dataset.num_features, dataset.num_classes, derivative=args.derivative, device=device).to(device)
else:
    raise Exception("Not model in list of models")

print(model)
print(TRAIN_SPLIT," ",BATCH_SIZE," " ,dataset.num_classes," ",dataset.num_features)
print(dataset[0].x)
######################################################
train_log_file = "ComplexDerivativeTEST_"
#RandList = [12345, 42345, 64345, 54345, 74345, 47345, 54321, 14321, 94321, 84328]
#RandList = RandList[:]

#DERIVATIVE = "laplacian" #laplacian or normalized

#BATCH_SIZE = 100
#DATASET = 'CIFAR10'
#TRAIN_SPLIT = 50000 if DATASET=='MNIST' else 40000


train_log_file = train_log_file + args.dataset +time.strftime('%d_%m_%y__%H_%M') + '.txt'

######################################################
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

#dataset = GNNBenchmarkDataset(root='data/GNNBenchmarkDataset', name=DATASET) #MNISTo CIFAR10
#data = dataset[0]  # Get the first graph object.
#dataset = dataset.shuffle()

train_dataset = dataset[:TRAIN_SPLIT] #MNIST : 50000 - CIFAR: 40000
test_dataset = dataset[TRAIN_SPLIT:] 
#torch.autograd.set_detect_anomaly(True)
ExperimentResult = []

f = open(train_log_file, 'w') #clear file
f.close()
for e in range(len(RandList)):
    #Da problemas el shuffle con los dataset sin features
    #dataset = dataset.shuffle()
    train_dataset = dataset[:TRAIN_SPLIT] #MNIST : 50000 - CIFAR: 40000
    test_dataset = dataset[TRAIN_SPLIT:] 
    #model = GAPNet(dataset.num_features, dataset.num_classes, derivative=DERIVATIVE, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)#
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Original 64
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Original 64
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
