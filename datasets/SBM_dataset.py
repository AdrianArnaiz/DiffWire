import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.random import stochastic_blockmodel_graph
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

class SBM_pyg(InMemoryDataset):
    def __init__(self, root, nb_nodes1=200, nb_graphs1=500, nb_nodes2=200, nb_graphs2=500, p1=0.8, p2=0.5, qmin1=0.5, qmax1=0.8, qmin2=0.25, qmax2=0.7, directed=False, transform=None, pre_transform=None):
        """
        Create SBM dataset with graph of 2 classes. Each graph will have 2 communities.
        For each class we can parametrize the number of nodes, number of graphs and
            intra-interclass edge probability max an min probability.

        Args:
            root (str): path to save the data.
            nb_nodes1 (int, optional): number of nodes in the graphs of class 1. Defaults to 200.
            nb_graphs1 (int, optional): number of graphs of class 1. Defaults to 500.
            nb_nodes2 (int, optional): number of nodes in the graphs of class 2. Defaults to 200.
            nb_graphs2 (int, optional):  number of graphs of class 2. Defaults to 500.
            p1 (float, optional): intraclass edge probability for community 1 for both graph classes. Defaults to 0.8.
            p2 (float, optional): intraclass edge probability for community 2 for both graph classes. Defaults to 0.5.
            qmin1 (float, optional): minimun intercalass probability for graphs in class 1. Defaults to 0.5.
            qmax1 (float, optional):  minimun interclass probability for graphs in class 2. Defaults to 0.8.
            qmin2 (float, optional): maximun intercalass probability for graphs in class 1. Defaults to 0.25.
            qmax2 (float, optional): maximun intercalass probability for graphs in class 2. Defaults to 0.7.
            directed (bool, optional): Create directed or Undirected Graphs. Defaults to False.
            transform (torch_geometric.transforms.BaseTransform, optional): on the fly transformation. Defaults to None.
            pre_transform (torch_geometric.transforms.BaseTransform, optional): transformation to save in disk. Defaults to None.
        """
        self.nb_nodes1 = nb_nodes1
        self.nb_graphs1 = nb_graphs1
        self.nb_nodes2 = nb_nodes2
        self.nb_graphs2 = nb_graphs2
        self.nb_graphs = self.nb_graphs1 + self.nb_graphs2
        #self.num_features = 1 # Degree
        #self.num_classes = 2
        self.edge_probs1_min = [[p1, qmin1], [qmin1, p2]] # Minimal setting class 1
        self.edge_probs2_min = [[p1, qmin2], [qmin2, p2]] # Minimal setting class 2
        self.edge_probs1_max = [[p1, qmax1], [qmax1, p2]] # Maximal setting class 1
        self.edge_probs2_max = [[p1, qmax2], [qmax2, p2]] # Maximal setting class 2

        self.details = f"{self.nb_nodes1}nodesT{self.nb_graphs}graphsT{int(p1*100)}" \
                        + f"p1T{int(p2*100)}p2T{int(qmin1*100)}to{int(qmax1*100)}q1T{int(qmin2*100)}to{int(qmax2*100)}q2"
        self.root = f"{root}_{self.details}"
        print(self.root)
        super(SBM_pyg, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['tentative']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` lists.
        data_list1 = self._generate_graphs(self.nb_nodes1, self.nb_graphs1, self.edge_probs1_min, self.edge_probs1_max, 0)
        data_list2 = self._generate_graphs(self.nb_nodes2, self.nb_graphs2, self.edge_probs2_min, self.edge_probs2_max, 1)
        data_list = [*data_list1, *data_list2]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_graphs(self, nb_nodes, nb_graphs, edge_probs_min, edge_probs_max, myclass):
        # p and q are static 
        # qmin < qmax 
        # for each graph move from qmin to qmax
        qmin = edge_probs_min[1][0]
        qmax = edge_probs_max[1][0]
        m  = (qmax - qmin)/nb_graphs # Linear slope
        dataset = []
        for i in range(nb_graphs):
            q = m*(i+1) + qmin
            p1 = edge_probs_min[0][0]
            p2 = edge_probs_min[1][1]
            # Get the SBM graph
            d = stochastic_blockmodel_graph(block_sizes=[int(nb_nodes/2),int(nb_nodes/2)], edge_probs=[[p1, q],[q, p2]], directed=False)

            # Get degree as feature 
            adj = to_dense_adj(d)
            A = adj
            D = A.sum(dim=1)
            x = torch.transpose(D,0,1)
            mydata = Data(x=x, edge_index=d, y = myclass, num_nodes=nb_nodes)
            dataset.append(mydata)
        return dataset

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.details})'


if __name__ == "__main__":
    print("Tesdting SBM")
    dataset = SBM_pyg('../data/SBM_final', nb_nodes1=200, nb_graphs1=500, nb_nodes2=200, nb_graphs2=500, p1=0.8, p2=0.5, qmin1=0.5, qmax1=0.8, qmin2=0.25, qmax2=0.71, directed=False, transform=None, pre_transform=None)

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(dataset.details)
    
    """train_dataset = dataset[:int(0.8*len(dataset))]
    test_dataset = dataset[int(0.8*len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Original 64
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Original 64
    

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()"""

