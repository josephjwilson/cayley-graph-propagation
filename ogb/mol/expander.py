from collections import deque
from typing import Dict
import numpy as np
from primefac import primefac
from dataclasses import dataclass
from enum import Enum
import torch
import torch_geometric.utils as geom_utils
import hashlib

def cayley_graph_size(n):
    """
        Calculate the number of nodes in the Cayley graph (Cay(SL(2, Z_n); S_n)).
    """
    n = int(n)
    return round(n*n*n*np.prod([1 - 1.0/(p * p) for p in list(set(primefac(n)))]))


def get_suitable_cayley_n(num_nodes, prime_only=False):
    """
        Get a suitable natural number `n` such that the number of nodes of the Cayley graph (Cay(SL(2, Z_n); S_n)) is at least as big as the target `num_nodes`.
        The parameter `prime_only` specifies whether `n` should be a prime number.
    """
    n = 1
    while cayley_graph_size(n) < num_nodes or (prime_only and len(list(primefac(n))) != 1):
        n += 1
    return n

def get_cayley_graph(n):
    """
        Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n)).
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
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
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
            ind_tx = nodes[tx_flat]

            senders.append(ind_x)
            receivers.append(ind_tx)
    return torch.tensor([senders, receivers])

class ShuffleType(Enum):
    NONE = 0
    RANDOM = 1  # Shuffle after taking memoized value

class ExpanderKind(Enum):
    NONE = -1
    EGP = 0
    CGP = 1

@dataclass
class ExpanderConfig:
    kind: ExpanderKind = ExpanderKind.NONE
    shuffle: ShuffleType = ShuffleType.NONE
    zero_virtual_node_embeddings: bool | None = None
    prime_only: bool = False
    
    @staticmethod
    def get_preset(name: str):
        return EXPANDER_CONFIGS[name]

    @staticmethod
    def get_preset_names():
        return list(EXPANDER_CONFIGS.keys())

    def __post_init__(self):
        if self.zero_virtual_node_embeddings is not None:
            assert self.kind is ExpanderKind.CGP, '`zero_virtual_node_embeddings` is only supported for CGP'

    def is_cgp(self):
        return self.kind is ExpanderKind.CGP

    def is_cgp_with_zeroed_virtual_node_embeddings(self):
        return self.kind is ExpanderKind.CGP and self.zero_virtual_node_embeddings
    
    def uses_expander_layers(self):
        return self.kind is not ExpanderKind.NONE
    
    def hash(self) -> str:
        string = str(self).encode('utf-8')
        return hashlib.md5(string).hexdigest()

EXPANDER_CONFIGS = {
    'none': ExpanderConfig(kind=ExpanderKind.NONE),
    'egp-base': ExpanderConfig(kind=ExpanderKind.EGP, prime_only=True),
    'cgp-base': ExpanderConfig(kind=ExpanderKind.CGP, zero_virtual_node_embeddings=True, prime_only=True),
    'egp-an': ExpanderConfig(kind=ExpanderKind.EGP, prime_only=False),
    'cgp-an': ExpanderConfig(kind=ExpanderKind.CGP, zero_virtual_node_embeddings=True, prime_only=False),
    'cgp-an-nzvne': ExpanderConfig(kind=ExpanderKind.CGP, zero_virtual_node_embeddings=False, prime_only=False),
    'egp-an-shfl': ExpanderConfig(kind=ExpanderKind.EGP, shuffle=ShuffleType.RANDOM, prime_only=True),
    'cgp-an-shfl': ExpanderConfig(kind=ExpanderKind.CGP, zero_virtual_node_embeddings=True, shuffle=ShuffleType.RANDOM, prime_only=True),
    'cgp-an-shfl-nzvne': ExpanderConfig(kind=ExpanderKind.CGP, zero_virtual_node_embeddings=False, shuffle=ShuffleType.RANDOM, prime_only=True),
}

assert len(set(config.hash() for config in EXPANDER_CONFIGS.values())) == len(EXPANDER_CONFIGS)

class PreTransform:
    def __init__(self, config: ExpanderConfig):
        self.config = config
        self.memory: Dict[int, torch.Tensor] = {}

    def shuffle_edge_index_inplace(self, num_nodes, edge_index):
        # Shuffle indices to remove potential bias from original dataset node ordering
        perm = np.random.permutation(num_nodes)
        for i in range(edge_index.shape[1]):
            edge_index[0][i] = perm[edge_index[0][i]]
            edge_index[1][i] = perm[edge_index[1][i]]


    def get_egp_edge_index(self, num_nodes, cayley_n):
        if num_nodes not in self.memory:
            edge_index = get_cayley_graph(cayley_n)
            truncated_edge_index = edge_index[:, torch.logical_and(edge_index[0] < num_nodes, edge_index[1] < num_nodes)]
            self.memory[num_nodes] = truncated_edge_index
        
        edge_index = self.memory[num_nodes].clone()
        if self.config.shuffle == ShuffleType.RANDOM:
            self.shuffle_edge_index_inplace(num_nodes, edge_index)
        return edge_index

    def get_cgp_edge_index(self, cayley_n):
        cayley_num_nodes = cayley_graph_size(cayley_n)
        if cayley_num_nodes not in self.memory:
            edge_index = get_cayley_graph(cayley_n)
            self.memory[cayley_num_nodes] = edge_index
        
        edge_index = self.memory[cayley_num_nodes].clone()
        if self.config.shuffle == ShuffleType.RANDOM:
            self.shuffle_edge_index_inplace(cayley_num_nodes, edge_index)
        return edge_index, cayley_num_nodes

    def __call__(self, datapoint):
        if self.config.kind is ExpanderKind.NONE:
            return datapoint
        
        store = next(store for store in datapoint.stores if 'edge_index' in store)
        num_nodes = store['num_nodes']
        suitable_cayley_n = get_suitable_cayley_n(num_nodes, prime_only=self.config.prime_only)
        
        if self.config.kind is ExpanderKind.EGP:
            store['expander_edge_index'] = self.get_egp_edge_index(num_nodes, suitable_cayley_n)
            store['expander_edge_attr'] = torch.zeros(
                size=(store['expander_edge_index'].shape[1], store['edge_attr'].shape[1]), 
                dtype=store['x'].dtype
            )
        elif self.config.kind is ExpanderKind.CGP:
            store['expander_edge_index'], cayley_num_nodes = self.get_cgp_edge_index(suitable_cayley_n)
            store['expander_edge_attr'] = torch.zeros(
                size=(store['expander_edge_index'].shape[1], store['edge_attr'].shape[1]), 
                dtype=store['x'].dtype
            )
            
            virtual_num_nodes = cayley_num_nodes - num_nodes
            store['virtual_node_mask'] = torch.cat([torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)], axis=0)
            store['num_nodes'] += virtual_num_nodes
            
            store['x'] = torch.cat([store['x'], torch.zeros(size=(virtual_num_nodes, store['x'].shape[1]), dtype=store['x'].dtype)], axis=0)
            virtual_node_idx = torch.arange(num_nodes, cayley_num_nodes)
            virtual_edge_index = torch.stack([virtual_node_idx, virtual_node_idx])
            store['edge_index'] = torch.cat([store['edge_index'], virtual_edge_index], axis=1)
            store['edge_attr'] = torch.cat([store['edge_attr'], torch.zeros(size=(virtual_num_nodes, store['edge_attr'].shape[1]), dtype=store['edge_attr'].dtype)], axis=0)
        else:
            raise ValueError(f'Unknown expander kind: {self.config.kind}')
        return datapoint
