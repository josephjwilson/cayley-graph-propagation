import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

import math
import numpy as np
from collections import deque

_CAYLEY_BOUNDS = [
    (6, 2),
    (24, 3),
    (120, 5),
    (336, 7),
    (1320, 11),
    (2184, 13),
    (4896, 17),
    (6840, 19),
    (12144, 23),
    (24360, 29),
    (29760, 31),
    (50616, 37),
]

def build_cayley_bank():
    ret_edges = []

    for _, p in _CAYLEY_BOUNDS:
        generators = np.array([
            [[1, 1], [0, 1]],
            [[1, p-1], [0, 1]],
            [[1, 0], [1, 1]],
            [[1, 0], [p-1, 1]]])
        ind = 1

        queue = deque([np.array([[1, 0], [0, 1]])])
        nodes = {(1, 0, 0, 1): 0}

        senders = []
        receivers = []

        while queue:
            x = queue.pop()
            x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
            assert x_flat in nodes
            ind_x = nodes[x_flat]
            for i in range(4):
                tx = np.matmul(x, generators[i])
                tx = np.mod(tx, p)
                tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
                if tx_flat not in nodes:
                    nodes[tx_flat] = ind
                    ind += 1
                    queue.append(tx)
                ind_tx = nodes[tx_flat]

                senders.append(ind_x)
                receivers.append(ind_tx)

        ret_edges.append((p, [senders, receivers]))

    return ret_edges

# Here is where the changes to CGP start!
# Note: this is extremely rough, hence I would probably refactor it a lot to split the concerns between responsibilities
# This would be also nice to think about if we ever wanted to open-source our code :)
def batched_augment_cayley(num_graphs, batch, cayley_bank):
    node_lims = np.zeros(num_graphs)
    mappings = [[] for _ in range(num_graphs)]
    fake_mappings = [[] for _ in range(num_graphs)]

    for i in range(len(batch)):
        node_lims[batch[i]] += 1
        mappings[batch[i]].append(i)
        fake_mappings[batch[i]].append(i)

    senders = []
    receivers = []

    # I probably should change th word fake, however this is the 'virtual nodes'
    fake_senders = []
    fake_receivers = []

    # Track the original index
    og_index = len(batch)

    # This is a bad name, this basically keeps track of the index of the new upper node.
    start_index = len(batch)

    # The general idea is that it handles the input graph as always up to |V|
    # Outside the region of |V| are where the new virtual nodes begin
    # I.e. Say the nodes go up to 100 for the entire batch
    # A virtual node may be between 101-110 for the first graph in the batch
    # Then the second batch may between 110-120 etc.

    for g in range(num_graphs):
        p = 2
        chosen_i = -1
        for i in range(len(_CAYLEY_BOUNDS)):
            sz, p = _CAYLEY_BOUNDS[i]
            if sz >= node_lims[g]:
                chosen_i = i
                break

        assert chosen_i >= 0
        _p, edge_pack = cayley_bank[chosen_i]
        assert p == _p

        # Increment add extend the nodes to fill in the position of |V| > based on the prior upper limit
        if sz > int(node_lims[g]):
            missing_nodes = list(range(start_index, start_index + int(sz - node_lims[g])))
            fake_mappings[g].extend(missing_nodes)
            start_index += len(missing_nodes)

        r_mappings = mappings[g]
        f_mappings = fake_mappings[g]

        assert sz == len(f_mappings)
        
        # Here is the simple change to say do I align with the input graph or am I out of the range
        # If I'm out of the range - then I'm a virtual node
        # Note* I have a newer version that handles the input graph being offset
        # I.e. it does not align with node zero
        for v, w in zip(*edge_pack):
            if v < node_lims[g] and w < node_lims[g]:
                senders.append(r_mappings[v])
                receivers.append(r_mappings[w])
            else:
                fake_senders.append(f_mappings[v])
                fake_receivers.append(f_mappings[w])

    self_indexes_senders = []
    self_indexes_recievers = []
    for i in range(og_index, start_index):
        self_indexes_senders.append(i)
        self_indexes_recievers.append(i)

    senders.extend(fake_senders)
    receivers.extend(fake_receivers)

    # Add all  the new self edges
    for i in range(start_index):
        senders.append(i)
        receivers.append(i)

    edge_attr = []
    for i in range(len(senders)):
        edge_attr.append([0, 0, 0])  # Replace this with whatever edge features would be expected in your dataset :)

    # This output is a bit meh, and could be streamlined
    return [senders, receivers], edge_attr, start_index, [self_indexes_senders, self_indexes_recievers]

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        self.cayley_bank = build_cayley_bank()

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        num_graphs = batched_data.num_graphs
        num_nodes = x.shape[0]

        # This is similar to before, but with more outputs

        # self_edge_index are just the self connections, outside of the graph node range.
        # I.e. if it ends at 100, then [101, 101]...
        # From the batched Cayley graph this is outside of the range
        cayley_g, cayley_attr, max_node, self_edge_indexes = batched_augment_cayley(num_graphs, batch, self.cayley_bank)
        cayley_g = torch.LongTensor(cayley_g).cuda()
        cayley_attr = torch.LongTensor(cayley_attr).cuda()

        # New self edges for dummy nodes
        self_edge_indexes = torch.LongTensor(self_edge_indexes).cuda()
        # edges indexes to be be used that include real edges for the graph + self edges for dummy nodes
        new_edge_index = torch.cat((edge_index, self_edge_indexes), dim=1).cuda()

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]

        # Initialise them at layer 0 <- I just took the average nodes at this point
        h_list_zero = h_list[0]

        # Example of different strategies to initialise the dummy nodes
        # Create the dummy nodes, expanding on layer[0] average
        # average_real_nodes = torch.mean(h_list[0], dim=0).to(device)
        # dummy_nodes = torch.randn(max_node - num_nodes, self.emb_dim).to(device)

        # I have changed this to -1, 0
        dummy_nodes = torch.zeros(max_node - num_nodes, self.emb_dim).cuda()

        # Expand the dummy edge attr to match new_edge_index shape, of course we're just adding [0, 0, 0]
        # Set the first nodes to what they were + [0, 0, 0] for everything else
        dummy_edge_attr = torch.zeros(new_edge_index.shape[1], 3, dtype=torch.int64).cuda()
        dummy_edge_attr[:edge_attr.shape[0], :] = edge_attr

        # Expand the first layer to include dummy_nodes
        h_list[0] = torch.cat((h_list[0], dummy_nodes), dim=0)

        # The loop will now be larger than |V| containing the virtual nodes
        for layer in range(self.num_layer):
            
            # Alternate between Cayley graph and input graph
            if layer % 2 == 1:
                h = self.convs[layer](h_list[layer], cayley_g, cayley_attr)
            else:
                h = self.convs[layer](h_list[layer], new_edge_index, dummy_edge_attr)

            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        # Truncate back to the original input graph size for downstream usage
        node_representation = h_list[-1][:num_nodes]

        return node_representation

if __name__ == "__main__":
    pass
