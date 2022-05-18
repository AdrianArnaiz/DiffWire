from math import ceil
import os
import random

from sklearn.model_selection import train_test_split
from nets import CTNet, DiffWire, GAPNet, MinCutNet
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
import torch_geometric.transforms as T
from transform_features import FeatureDegree, DIGLedges
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import time
import argparse
import numpy as np
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

################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
        "--dataset",
        default="CIFAR10",
        choices=["MUTAG","ENZYMES","PROTEINS","CIFAR10","MNIST","COLLAB","IMDB-BINARY","REDDIT-BINARY","CSL"],
        help="nada",
)
parser.add_argument(
    "--model",
    default="CTNet",
    choices=["CTNet","GAPNet","MinCutNet", "DiffWire"],
    help="nada",
)
parser.add_argument(
    "--derivative",
    default="laplacian",
    choices=["laplacian","normalizedv2"], #,"normalized"
    help="nada",
)
parser.add_argument(
    "--cuda",
    default="cuda:0",
    choices=["cuda:0","cuda:1"],
    help="cuda version",
)
parser.add_argument(
    "--prepro",
    default=None,
    choices=[None,"digl"],
    help="digl preprocessing",
)
parser.add_argument(
    "--store",
    action="store_true",
    help="nada",
)
parser.add_argument(
    "--iter",
    type=int,
    default=10,
    help="The number of games to simulate"
    )
parser.add_argument(
    "--logs",
    default="logs",
    help="log folders",
)
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=1e-4, help="Outer weight decay rate of model"
    )
args = parser.parse_args()

#Procesing dataset
No_Features = ["COLLAB","IMDB-BINARY","REDDIT-BINARY", "CSL"]
preprocessing = DIGLedges(alpha=0.1) if args.prepro == "digl" else None
aux_digl_foler = "/DIGL" if args.prepro == "digl" else ""

if args.dataset not in GNNBenchmarkDataset.names:
    if args.dataset not in No_Features:
        dataset = TUDataset(root='data'+os.sep+aux_digl_foler+os.sep+'TUDataset', name=args.dataset, pre_transform=preprocessing)
        if args.dataset =="MUTAG": # 188 graphs
            TRAIN_SPLIT = 150
            BATCH_SIZE = 32
            num_of_centers = 17 #mean number of nodes according to PyGeom
        if args.dataset =="ENZYMES": # 600 graphs
            TRAIN_SPLIT = 500
            BATCH_SIZE = 32
            num_of_centers = 16 #mean number of nodes according to PyGeom
        if args.dataset =="PROTEINS": # 1113 graphs
            TRAIN_SPLIT = 1000
            BATCH_SIZE = 64
            num_of_centers = 39 #mean number of nodes according to PyGeom
    else: #Features
        if args.prepro == "digl":
            preprocessing = preprocessing
            processing = FeatureDegree()
        else:
            preprocessing = FeatureDegree()
            processing = None
        dataset = TUDataset(root='data'+os.sep+aux_digl_foler+os.sep+'TUDataset',name=args.dataset,
                            pre_transform=preprocessing, transform = processing, use_node_attr=True)
        #dataset = TUDatasetFeatures(root='data/TUDataset', name=args.dataset,dataset=datasetGNN)        
        if args.dataset =="IMDB-BINARY": # 1000 graphs
            TRAIN_SPLIT = 800
            BATCH_SIZE = 64
            num_of_centers = 20 #mean number of nodes according to PyGeom
        elif args.dataset == "REDDIT-BINARY":  # 2000 graphs
            TRAIN_SPLIT = 1500
            BATCH_SIZE = 64
            num_of_centers = 420 #mean number of nodes according to PyGeom
        elif args.dataset == "COLLAB":  # 2000 graphs
            TRAIN_SPLIT = 4500
            BATCH_SIZE = 64
            num_of_centers = 75 #mean number of nodes according to PyGeom
        else:
            raise Exception("Not dataset in list of datasets")
else: #GNNBenchmarkDataset
    if args.dataset  in No_Features:
        if args.prepro == "digl":
            preprocessing = preprocessing
            processing = FeatureDegree()
        else:
            preprocessing = FeatureDegree()
            processing = None

        dataset = GNNBenchmarkDataset(root='data'+os.sep+aux_digl_foler+os.sep+'GNNBenchmarkDataset', name=args.dataset,
                            pre_transform=preprocessing, transform = processing)
        if args.dataset =="CSL":
            TRAIN_SPLIT = 120
            BATCH_SIZE = 10
            num_of_centers = 42
    else:
        dataset = GNNBenchmarkDataset(root='data'+os.sep+aux_digl_foler+os.sep+'GNNBenchmarkDataset', name=args.dataset, pre_transform=preprocessing) #MNISTo CIFAR10
        if args.dataset =="MNIST":
            TRAIN_SPLIT = 50000
            BATCH_SIZE = 100
            num_of_centers = 100
        elif args.dataset == "CIFAR10":
            TRAIN_SPLIT = 40000
            BATCH_SIZE = 100
            num_of_centers = 100

##################### STATIC Variables #################################

device = args.cuda

N_EPOCH = 60

exp_name = f"{args.dataset}_{args.model}"
exp_name = exp_name+"DIGL" if args.prepro=="digl" else exp_name
exp_name = exp_name + f"_{args.derivative}" if args.model=="GAPNet" else exp_name
exp_time = time.strftime('%d_%m_%y__%H_%M')
train_log_file = exp_name + f"_{exp_time}.txt"

RandList = [12345, 42345, 64345, 54345, 74345, 47345, 54321, 14321, 94321, 84328]
RandList = RandList[:args.iter]

if not os.path.exists(args.logs):
    os.makedirs(args.logs)
if not os.path.exists("models/") and args.store:
    os.makedirs("models")

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
        pred, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        #print(next(model.parameters()).device)
        #print(data.x.device)
        loss = F.nll_loss(pred, data.y.view(-1)) + mc_loss + o_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)

######################################################
print(device)

#torch.autograd.set_detect_anomaly(True)
ExperimentResult = []

f = open(args.logs+os.sep+train_log_file, 'w') #clear file
print("- M:", args.model, "- D:",dataset,  
        "- Train_split:", TRAIN_SPLIT, "- B:",BATCH_SIZE,
        "- Centers (if CTNet):", num_of_centers, "- LAP (if GAPNet):", args.derivative,
        "- Classes" ,dataset.num_classes,"- Feats",dataset.num_features, file=f)
f.close()
EPS=1e-15
for e in range(len(RandList)):
    if args.model == 'CTNet':
        model = CTNet(dataset.num_features, dataset.num_classes, k_centers=num_of_centers, EPS=EPS).to(device)
    elif args.model == 'GAPNet':
        model = GAPNet(dataset.num_features, dataset.num_classes, derivative=args.derivative, device=device).to(device)
    elif args.model == 'MinCutNet':
        model = MinCutNet(dataset.num_features, dataset.num_classes).to(device)
    elif args.model == 'DiffWire':
        model = DiffWire(dataset.num_features, dataset.num_classes, k_centers=num_of_centers,
                        derivative="normalizedv2", device=device, EPS=1e-15).to(device)
    else:
        raise Exception(f"Not implemented model: {args.model}")
    
    train_indices, test_indices = train_test_split(list(range(len(dataset.data.y))), test_size=0.2, stratify=dataset.data.y,
                                    random_state=RandList[e], shuffle=True)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    

    print(len(train_dataset),len(test_dataset))
    #model = GAPNet(dataset.num_features, dataset.num_classes, derivative=DERIVATIVE, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  #
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

        f = open(args.logs+os.sep+train_log_file, 'a')
        print('Epoch: {:03d}, '
                'Train Loss: {:.3f}, Train Acc: {:.3f}, '
                'Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, train_loss,
                                                            train_acc, test_loss,
                                                            test_acc), file=f)
        print('{} - Epoch: {:03d}, '
                'Train Loss: {:.3f}, Train Acc: {:.3f}, '
                'Test Loss: {:.3f}, Test Acc: {:.3f}, Time: {:.2f}'.format(exp_name, epoch, train_loss,
                                                            train_acc, test_loss,
                                                            test_acc, time_lapse))
        f.close()

    if args.store:
        torch.save(model.state_dict(), f"models{os.sep}{exp_name}_{exp_time}_iter{e}.pth")
        print(f"Model saved in models{os.sep}{exp_name}_{exp_time}_iter{e}.pth")

    ExperimentResult.append(test_acc)
    f = open(args.logs+os.sep+train_log_file, 'a')
    print('Result of run {:.3f} is {:.3f}'.format(e,test_acc), file=f)
    f.close()

f = open(args.logs+os.sep+train_log_file, 'a')
print('Test Acc of 10 execs {}'.format(ExperimentResult), file=f)
print('{} +- {}'.format(np.mean(ExperimentResult), np.std(ExperimentResult)), file=f)
f.close()
