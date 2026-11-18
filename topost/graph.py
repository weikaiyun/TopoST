"""Disjoint METIS units and deterministic structural node descriptors."""
from dataclasses import dataclass
import math
import numpy as np
import torch


def structural_descriptors(adjacency):
    """Sixteen statistical features, invariant to node labels.

    Columns: log-degree, log-strength; log mean/std/min/max neighbor degree;
    log mean/std neighbor strength; clustering, mean neighbor clustering;
    log two/three-hop neighborhood size; ego density; edge reciprocity;
    log mean/std incident weight. Empty neighborhoods have zero summaries.
    """
    adjacency = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(adjacency, 0)
    symmetric = (adjacency + adjacency.T) / 2
    support = symmetric > 0
    degree = support.sum(1).astype(float)
    strength = symmetric.sum(1)
    binary = support.astype(float)
    triangles_twice = np.diag(binary @ binary @ binary)
    clustering = np.divide(triangles_twice, degree * (degree - 1),
                           out=np.zeros_like(degree), where=degree > 1)
    reach = np.eye(len(adjacency), dtype=bool) | support
    two_hop = (reach.astype(int) @ reach.astype(int)) > 0
    three_hop = (two_hop.astype(int) @ reach.astype(int)) > 0
    features = np.zeros((len(adjacency), 16), dtype=np.float32)
    features[:, 0] = np.log1p(degree)
    features[:, 1] = np.log1p(strength)
    features[:, 8] = clustering
    features[:, 10] = np.log1p(two_hop.sum(1) - 1)
    features[:, 11] = np.log1p(three_hop.sum(1) - 1)
    for i, neighbors in enumerate(support):
        if not neighbors.any():
            continue
        values = degree[neighbors]
        features[i, 2:6] = np.log1p([values.mean(), values.std(), values.min(), values.max()])
        values = strength[neighbors]
        features[i, 6:8] = np.log1p([values.mean(), values.std()])
        features[i, 9] = clustering[neighbors].mean()
        ego = neighbors.copy()
        ego[i] = True
        size = int(ego.sum())
        features[i, 12] = support[np.ix_(ego, ego)].sum() / (size * (size - 1))
        features[i, 13] = ((adjacency[i] > 0) & (adjacency[:, i] > 0)).sum() / degree[i]
        weights = symmetric[i, neighbors]
        features[i, 14:16] = np.log1p([weights.mean(), weights.std()])
    return features


@dataclass
class Graph:
    nodes: torch.Tensor
    support: torch.Tensor
    fixed: torch.Tensor
    features: torch.Tensor
    mask: torch.Tensor

    def to(self, device):
        return Graph(*(getattr(self, key).to(device) for key in
                       ('nodes', 'support', 'fixed', 'features', 'mask')))


def make_graph(adjacency, nodes):
    """Directed weights for forecasting; binary symmetric support for GIN."""
    local = np.asarray(adjacency[np.ix_(nodes, nodes)], dtype=np.float64).copy()
    np.fill_diagonal(local, 0)
    support = ((local + local.T) > 0).astype(np.float32)
    loop = local + np.eye(len(nodes))
    inverse_degree = 1 / np.sqrt(loop.sum(1))
    fixed = inverse_degree[:, None] * loop * inverse_degree[None, :]
    return Graph(torch.as_tensor(nodes, dtype=torch.long), torch.from_numpy(support),
                 torch.as_tensor(fixed, dtype=torch.float32),
                 torch.from_numpy(structural_descriptors(local)),
                 torch.ones((len(nodes), len(nodes)), dtype=torch.bool))


def partition_graph(adjacency, core_size=16, seed=42):
    adjacency = np.asarray(adjacency, dtype=np.float64).copy()
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or not len(adjacency) or not np.isfinite(adjacency).all()
            or (adjacency < 0).any() or core_size < 1):
        raise ValueError('Expected nonnegative finite square adjacency and positive core size.')
    np.fill_diagonal(adjacency, 0)
    num_parts = math.ceil(len(adjacency) / core_size)
    if num_parts == 1:
        partitions = [np.arange(len(adjacency))]
    else:
        import pymetis
        support = (adjacency > 0) | (adjacency.T > 0)
        neighbors = [np.flatnonzero(row).tolist() for row in support]
        membership = np.asarray(pymetis.part_graph(
            num_parts, adjacency=neighbors, options=pymetis.Options(seed=seed))[1])
        partitions = [np.flatnonzero(membership == i) for i in range(num_parts)]
        partitions = [nodes for nodes in partitions if len(nodes)]
    covered = np.concatenate(partitions)
    if not np.array_equal(np.sort(covered), np.arange(len(adjacency))):
        raise ValueError('METIS must cover each node exactly once.')
    return [make_graph(adjacency, nodes) for nodes in partitions]


def pack_graphs(graphs, max_nodes=128):
    """Pack whole units; the mask prevents adaptive edges between units."""
    if max_nodes < 0:
        raise ValueError('max_nodes must be nonnegative; zero means no limit.')
    groups, group, count = [], [], 0
    for graph in graphs:
        size = len(graph.nodes)
        if max_nodes and size > max_nodes:
            raise ValueError('max_nodes is smaller than one complete METIS unit.')
        if max_nodes and group and count + size > max_nodes:
            groups.append(group)
            group, count = [], 0
        group.append(graph)
        count += size
    if group:
        groups.append(group)
    return [Graph(torch.cat([g.nodes for g in group]),
                  torch.block_diag(*[g.support for g in group]),
                  torch.block_diag(*[g.fixed for g in group]),
                  torch.cat([g.features for g in group]),
                  torch.block_diag(*[g.mask for g in group])) for group in groups]
