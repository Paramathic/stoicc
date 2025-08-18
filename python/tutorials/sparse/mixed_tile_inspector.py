# Sample Sparse Inspector
#  This file provides sample code for a sparse inspector that takes a dense matrix, a target tile
#  shape, and a number of sparse tiles to assign. It then chooses tiles to be sparse at random.
#  Expand this file as needed to apply and experiment with new sparse inspection strategies.

import torch

import random
import numpy as np
from enum import Enum
from dataclasses import dataclass

from prune import prune_2_4
from compress import compress_dense_to_sparse
from triton.sparsity.compressed_sparse import CompressedSparse

@dataclass
class MMAShape:
    M : int
    N : int
    K : int
    m : int
    n : int
    k : int

    num_warps  : int
    num_stages : int
    group_size : int

    sparseA : bool = True

    def __post_init__(self):
        if self.sparseA:
            self.tiles_row = self.M // self.m
            self.tiles_col = self.K // self.k
        else:
            self.tiles_row = self.N // self.n
            self.tiles_col = self.K // self.k

class Sparsity(Enum):
    NV_24 = 0 # 2:4 sparse tile
    DENSE = 1 # Fully dense tile
    EMPTY = 2 # All-zero tile

def pT(x):
    '''
    Transpose and return X, make it contiguous
    '''
    return x.transpose(0, 1).contiguous()

def generate_sparsity_pattern(shape, n_sparse, n_empty, is_global):
    """
    Randomly assign a sparsity to each tile.
    If `is_global` is not set, will assign `n_sparse` per row, otherwise globally.
    """
    if is_global:
        # Generate a sparsity map, with `n_sparse` total sparse tiles

        assert(shape.tiles_row <= n_sparse <= shape.tiles_row * shape.tiles_col)

        pattern = np.full((shape.tiles_row, shape.tiles_col), Sparsity.DENSE.value)

        # Initially assign one sparse tile per row
        for i in range(shape.tiles_row):
            pattern[i][random.randint(0, shape.tiles_col - 1)] = Sparsity.NV_24.value

        placed = shape.tiles_row

        # Assign remainder of tiles
        while placed < n_sparse:
            r = random.randint(0, shape.tiles_row - 1)
            c = random.randint(0, shape.tiles_col - 1)

            if pattern[r][c] != Sparsity.NV_24.value:
                pattern[r][c] = Sparsity.NV_24.value
                placed += 1

        placed = 0

        while placed < n_empty:
            r = random.randint(0, shape.tiles_row - 1)
            c = random.randint(0, shape.tiles_col - 1)

            if pattern[r][c] not in (Sparsity.NV_24.value, Sparsity.EMPTY.value):
                pattern[r][c] = Sparsity.EMPTY.value
                placed += 1

    else:
        # Generate a sparsity map, with `per_row` sparse tiles per row
        # If per_row is -1, generate a random number of sparse tiles per row

        assert(n_sparse <= shape.tiles_col)
        assert(n_empty == 0)

        def get_col():
            n = random.randint(1, shape.tiles_row) if n_sparse < 0 else n_sparse
            return np.random.permutation(n * [Sparsity.NV_24.value] + (shape.tiles_col - n) * [Sparsity.DENSE.value])
        pattern = np.array([get_col() for _ in range(shape.tiles_row)])

    return pattern

def create_mixed_sparsity(A, sparsity_map):
    # Given a matrix and a sparsity map, performs reordering and scheduling.

    rows, cols = len(sparsity_map), len(sparsity_map[0])
    d0, d1 = A.shape[0] // rows, A.shape[1] // cols

    # Calculate the switches needed to bring the 2:4 elements before the dense elements for every row
    mapping = np.argsort(sparsity_map, kind='stable')
    row_diffs = np.diff(mapping, prepend=0)
    row_counts = np.apply_along_axis(lambda x: (x == Sparsity.NV_24.value).sum(), 1, sparsity_map)
    row_counts_dense = np.apply_along_axis(lambda x: (x == Sparsity.DENSE.value).sum(), 1, sparsity_map)
    sparse_row_offsets = np.concatenate(([0],  np.cumsum(row_counts)))
    dense_row_offsets = np.concatenate(([0],  np.cumsum(row_counts_dense)))

    def store(T, ti, tj, F, fi, fj, prune = None):
        if prune is None or prune == Sparsity.DENSE.value:
            T[d0 * ti : d0 * (ti + 1), d1 * tj : d1 * (tj + 1)] = \
                F[d0 * fi : d0 * (fi + 1), d1 * fj : d1 * (fj + 1)]
        elif prune == Sparsity.NV_24.value:
            T[d0 * ti : d0 * (ti + 1), d1 * tj : d1 * (tj + 1)] = \
                prune_2_4(F[d0 * fi : d0 * (fi + 1), d1 * fj : d1 * (fj + 1)])
        elif prune == Sparsity.EMPTY.value:
            T[d0 * ti : d0 * (ti + 1), d1 * tj : d1 * (tj + 1)] = 0
        else:
            raise Exception("Unknown pruning method.")

    # Prune A per tile
    for i in range(rows):
        for j in range(cols): store(A, i, j, A, i, j, prune = sparsity_map[i][j])

    # Decompose into sparse and dense buffers
    total_sparse = np.sum(row_counts)
    total_dense = np.sum(row_counts_dense)
    s = torch.zeros((d0, d1 * total_sparse))
    d = torch.zeros((d0, d1 * total_dense))
    si, di = 0, 0
    for i in range(rows):
        n_sparse = row_counts[i]
        n_dense = row_counts_dense[i]
        for j in range(cols):
            if j < n_sparse:
                store(s, 0, si, A, i, mapping[i][j])
                si += 1
            elif j < (n_sparse + n_dense):
                store(d, 0, di, A, i, mapping[i][j])
                di += 1

    return A, s, d, row_diffs, sparse_row_offsets, dense_row_offsets

def inspect_tiled(shape, sparse, empty, keep = False):
    # Perform the inspection phase.

    if shape.sparseA:
        y = torch.randn(shape.K, shape.N)

        pat = generate_sparsity_pattern(shape, sparse, empty, is_global=True)
        x, x_sparse, x_dense, row_diffs, sparse_row_offsets, dense_row_offsets = \
            create_mixed_sparsity(torch.randn(shape.M, shape.K), pat)

        x_sparse = x_sparse.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)
        x_dense = x_dense.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)
        x_compressed = CompressedSparse.NV24(*compress_dense_to_sparse(x_sparse))

        if keep:
            x = x.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)

        y = y.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)

        row_diffs = torch.Tensor(row_diffs).contiguous().to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)
        sparse_row_offsets = torch.Tensor(sparse_row_offsets).contiguous().to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)
        dense_row_offsets = torch.Tensor(dense_row_offsets).contiguous().to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)

        torch.cuda.synchronize()

        if keep:
            return (x, y), (x_compressed, x_dense, y, row_diffs, sparse_row_offsets, dense_row_offsets)
        else:
            return x_compressed, x_dense, y, row_diffs, sparse_row_offsets, dense_row_offsets
    else:
        x = torch.randn(shape.M, shape.K)

        pat = generate_sparsity_pattern(shape, sparse, empty, is_global=True)
        y, y_sparse, y_dense, row_diffs, sparse_row_offsets, dense_row_offsets = \
                create_mixed_sparsity(torch.randn(shape.N, shape.K), pat)

        y_sparse = y_sparse.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)
        y_dense = y_dense.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)
        y_compressed = CompressedSparse.NV24(*compress_dense_to_sparse(y_sparse))

        if keep:
            y = y.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)

        x = x.pin_memory().to(device="cuda:0", dtype=torch.float16, copy=True, non_blocking=False)

        row_diffs = torch.Tensor(row_diffs).to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)
        sparse_row_offsets = torch.Tensor(sparse_row_offsets).contiguous().to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)
        dense_row_offsets = torch.Tensor(dense_row_offsets).contiguous().to(device="cuda:0", dtype=torch.int32, copy=True, non_blocking=False)

        torch.cuda.synchronize()

        if keep:
            return (x, pT(y)), (x, y_compressed, y_dense, row_diffs, sparse_row_offsets, dense_row_offsets)
        else:
            return x, y_compressed, y_dense, row_diffs, sparse_row_offsets, dense_row_offsets