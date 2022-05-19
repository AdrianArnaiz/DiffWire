from sklearn.model_selection import train_test_split
import torch
from nets import CTNet, GAPNet
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from transform_features import FeatureDegree
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from embeddings_readout_utils import CTNet_readout_embedding, GAPNet_readout_embedding, MinCutNet_readout_embedding
from embeddings_readout_utils import print_diff_readout_embeddings, test_readout_embedd, print_readout_embeddings
import numpy as np
import time 

###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:0"

print(device)

model_family = 'GAP' # CT, GAP, MinCut
model_path = "models/REDDIT-BINARY_GAPNet_normalized_19_05_22__10_09_iter0.pth"
#model_family = 'MinCut' # CT, GAP, MinCut
#model_path = "trained_models/REDDIT-BINARY_MinCutNet_16_05_22__11_05_iter0.pth"
#model_family = 'CT' # CT, GAP, MinCut
#model_path = "trained_models/REDDIT-BINARY_CTNet_17_05_22__08_50_iter0.pth"

dataset = 'REDDIT' #REDDIT, COLLAB, IMDB

SAVE_PATH = "figs/"+model_family+'_'+dataset+"_READOUT_"+time.strftime('%d_%m_%y__%H_%M')

fig_style = 'wrong_pred' #["real_predicted",  "wrong_pred"]
####################

###### Datasets with same parameters and order thatn in training.
dataset = TUDataset(root='data_colab/TUDataset',name="REDDIT-BINARY", pre_transform=FeatureDegree(), use_node_attr=True)
BATCH_SIZE = 64 #REDDIT - dependeing on memory
num_of_centers = 420

#Same order than training
train_indices, test_indices = train_test_split(list(range(len(dataset.data.y))), test_size=0.2, stratify=dataset.data.y,
                                random_state=12345, shuffle=True)
train_indices.extend(test_indices)
new_order = train_indices
dataset = dataset[new_order]
loader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(dataset)

###### Model that throws readout embeddings

if model_family=='CT':
    model =  CTNet_readout_embedding(dataset.num_features, dataset.num_classes, k_centers=num_of_centers).to(device)
elif model_family == 'MinCut':
    model =  MinCutNet_readout_embedding(dataset.num_features, dataset.num_classes).to(device)
elif model_family == 'GAP':
    model = GAPNet_readout_embedding(dataset.num_features, dataset.num_classes, derivative='laplacian', device=device).to(device)
else:
    raise Exception("Not implemented method")


model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()

loss, acc, embeddings, labels, predictions = test_readout_embedd(model, loader, device)
embeddings = embeddings.detach().cpu()

print('Embedding shape:',embeddings.shape, embeddings.device)
print('Dataset size:',len(predictions))
print('Test loss and acc:',loss, acc)

e = np.array(embeddings)
p = np.array(predictions)
l = np.array(labels)
print('Array Acc:',(np.sum(p == l)/len(p))*100)
e_train = e[train_indices]
e_test  = e[test_indices]
lab_train = l[train_indices]
lab_test  = l[test_indices]
pred_train = p[train_indices]
pred_test  = p[test_indices]

if fig_style == "real_predicted":
    print_readout_embeddings(e_train, e_test, lab_train, lab_test, pred_train, pred_test, len(set(l)), 
                            save_path=SAVE_PATH, title=model_family)
elif fig_style == "wrong_pred":
    print_diff_readout_embeddings(e_train, e_test, lab_train, lab_test, pred_train, pred_test, len(set(l)), 
                            save_path=SAVE_PATH, title=model_family, seed=12345)

