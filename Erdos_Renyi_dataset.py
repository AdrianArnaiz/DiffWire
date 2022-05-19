
from torch_geometric.utils.random import erdos_renyi_graph
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

class Erdos_Renyi_pyg(InMemoryDataset):
    def __init__(self, root, nb_nodes1=200, nb_graphs1=500, nb_nodes2=200, nb_graphs2=500, p1_min=0.1, p1_max=0.5,
                p2_min=0.5, p2_max=0.8, directed=False, transform=None, pre_transform=None):
        self.nb_nodes1 = nb_nodes1
        self.nb_graphs1 = nb_graphs1
        self.nb_nodes2 = nb_nodes2
        self.nb_graphs2 = nb_graphs2
        self.nb_graphs = self.nb_graphs1 + self.nb_graphs2

        self.directed = directed
        
        self.p1_min = p1_min
        self.p1_max = p1_max
        self.p2_min = p2_min
        self.p2_max = p2_max

        self.details = f"{self.nb_nodes1}nodesT{self.nb_graphs}graphsT" \
                        + f"{int(p1_min*100)}to{int(p1_max*100)}p1T{int(p2_min*100)}to{int(p2_max*100)}p2"
        self.root = f"{root}_{self.details}"
        print(self.root)
        super(Erdos_Renyi_pyg, self).__init__(self.root, transform, pre_transform)
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
        data_list1 = self._generate_graphs(self.nb_nodes1, self.nb_graphs1, self.p1_min, self.p1_max, 0)
        data_list2 = self._generate_graphs(self.nb_nodes2, self.nb_graphs2, self.p2_min, self.p2_max, 1)
        data_list = [*data_list1, *data_list2]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_graphs(self, nb_nodes, nb_graphs, p_min, p_max, myclass):
        dataset = []
        m  = (p_max - p_min)/nb_graphs # Linear slope
        dataset = []
        for i in range(nb_graphs):
            p_i = m*(i+1) + p_min
            # Get the SBM graph
            d = erdos_renyi_graph(num_nodes=nb_nodes, edge_prob=p_i, directed=self.directed)

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
    print("Tesdting Erdos-Renyi")
    dataset = Erdos_Renyi_pyg('./data/SBM_final', nb_nodes1=200, nb_graphs1=500, nb_nodes2=200, nb_graphs2=500,
                        p1_min=0.1, p1_max=0.5, p2_min=0.3, p2_max=0.8)
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
    print(f'Label: {data.y}')
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

