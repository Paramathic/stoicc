import torch, triton
import triton.language as tl

import random
import numpy as np
from enum import Enum
from dataclasses import dataclass

from prune import prune_2_4
from compress import compress_dense_to_sparse
from triton.sparsity.compressed_sparse import CompressedSparse

from triton import testing

# TODO: Fix loop if loop no sparse tiles in row
# TODO: Autotuning
# TODO: B-sparse version

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

    def __post_init__(self):
        self.tiles_row = self.M // self.m
        self.tiles_col = self.K // self.k

class Sparsity(Enum):
    NV_24 = 0 # 2:4 sparse tile
    DENSE = 1 # Fully dense tile
    EMPTY = 2 # All-zero tile

# Parameters
SHAPE = MMAShape(4096, 4096, 4096, 256, 128, 64,
                 num_warps = 8, num_stages = 3, group_size = 8)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': SHAPE.m, 'BLOCK_SIZE_N': SHAPE.n, 'BLOCK_SIZE_K': SHAPE.k, 'GROUP_SIZE_M': SHAPE.group_size},
                      num_stages=SHAPE.num_stages, num_warps=SHAPE.num_warps)
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    a_ptrs = tl.make_block_ptr(a_ptr,
                               shape=(M, K),
                               strides=(stride_am, stride_ak),
                               offsets=(pid_m*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                               order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(K, N),
                               strides=(stride_bk, stride_bn),
                               offsets=(0, pid_n*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                               order=(1,0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        a = tl.load(a_ptrs, boundary_check=[1], padding_option="zero")
        b = tl.load(b_ptrs, boundary_check=[1], padding_option="zero")

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_SIZE_K])
        b_ptrs = tl.advance(b_ptrs, [BLOCK_SIZE_K, 0])
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M, N),
                               strides=(stride_cm, stride_cn),
                               offsets=(pid_m*BLOCK_SIZE_M, pid_n*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                               order=(1,0))
    tl.store(c_ptrs, c, boundary_check=(1,0))

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )

    return c

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tiled_kernel(
        # Pointers to matrices
        a_sparse_ptr, a_dense_ptr, a_col, a_sparse_row, a_dense_row,
        b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K, SK, DK,
        stride_asm, stride_ask,  #
        stride_adm, stride_adk,  #
        stride_bk, stride_bn,    #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Kernel for computing a tiled matmul with 2:4 sparse, dense, and all-zero tiles.
    - A is tiled, and each tile has an assigned sparsity type. B is dense.
    - Tiles of A stored in a csr-like format.
        - Sparse tiles stored in `a_sparse_ptr`
        - Dense tiles stored in `a_dense_ptr`
        - `a_sparse_row`, `a_dense_row`, and `a_col` contain row pointers and column indices
    - Currently, all rows need at least one sparse tile
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_row = first_pid_m + (pid % group_size_m)
    pid_col = (pid % num_pid_in_group) // group_size_m

    a_sparse_row += pid_row
    a_dense_row += pid_row
    sparse_l = tl.load(a_sparse_row)
    sparse_r = tl.load(a_sparse_row + 1)
    dense_l  = tl.load(a_dense_row)
    dense_r  = tl.load(a_dense_row + 1)

    a_col += pid_row * tl.cdiv(K, BLOCK_SIZE_K)
    first_b = tl.load(a_col)
    a_col += 1

    a_sparse_ptrs = tl.make_block_ptr(a_sparse_ptr,
                                      shape=(BLOCK_SIZE_M,SK), strides=(stride_asm, stride_ask), offsets=(0, sparse_l*BLOCK_SIZE_K),
                                      block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1,0))
    a_dense_ptrs = tl.make_block_ptr(a_dense_ptr,
                                     shape=(BLOCK_SIZE_M,DK), strides=(stride_adm, stride_adk), offsets=(0, dense_l*BLOCK_SIZE_K),
                                     block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(K,N), strides=(stride_bk, stride_bn), offsets=(first_b*BLOCK_SIZE_K, pid_col*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1,0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Sparse Loop
    for k in range(sparse_r - sparse_l):
        accumulator = tl.dot(
            tl.load(a_sparse_ptrs, boundary_check=[1], padding_option="zero"),
            tl.load(b_ptrs, boundary_check=[1], padding_option="zero"),
            accumulator
        )

        a_sparse_ptrs = tl.advance(a_sparse_ptrs, [0, BLOCK_SIZE_K])
        b_ptrs = tl.advance(b_ptrs, [tl.load(a_col) * BLOCK_SIZE_K, 0])
        a_col += 1

    # Dense Loop
    for k in range(dense_r - dense_l):
        accumulator = tl.dot(
            tl.load(a_dense_ptrs, boundary_check=[1], padding_option="zero"),
            tl.load(b_ptrs, boundary_check=[1], padding_option="zero"),
            accumulator
        )

        a_dense_ptrs = tl.advance(a_dense_ptrs, [0, BLOCK_SIZE_K])
        b_ptrs = tl.advance(b_ptrs, [tl.load(a_col) * BLOCK_SIZE_K, 0])
        a_col += 1

    c = accumulator.to(tl.float16)

    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M,N), strides=(stride_cm,stride_cn), offsets=(pid_row*BLOCK_SIZE_M, pid_col*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1,0))

    tl.store(c_ptrs, c, boundary_check=(1,0))

def tiled(a_sparse, a_dense, b, a_col, a_sparse_row, a_dense_row,):
    M, N, K = SHAPE.M, SHAPE.N, SHAPE.K
    SK, DK = a_sparse.shape[1], a_dense.shape[1]
    # Allocates output.
    c = torch.empty((M, N), device=b.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    tiled_kernel[grid](
        a_sparse, a_dense, a_col, a_sparse_row, a_dense_row, #
        b, c,  #
        M, N, K, SK, DK, #
        a_sparse.stride(0), a_sparse.stride(1),  #
        a_dense.stride(0), a_dense.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )

    return c

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

        def get_col():
            n = random.randint(1, shape.tiles_row) if n_sparse < 0 else n_sparse
            return np.random.permutation(n * [Sparsity.NV_24.value] + (shape.tiles_col - n) * [Sparsity.DENSE.value])
        pattern = np.array([get_col() for _ in range(shape.tiles_row)])

    return pattern

def create_mixed_sparsity(A, sparsity_map):
    rows, cols = len(sparsity_map), len(sparsity_map[0])
    m, k = A.shape[0] // rows, A.shape[1] // cols

    # Calculate the switches needed to bring the 2:4 elements before the dense elements for every row
    mapping = np.argsort(sparsity_map, kind='stable')
    row_diffs = np.diff(mapping, prepend=0)
    row_counts = np.apply_along_axis(lambda x: (x == Sparsity.NV_24.value).sum(), 1, sparsity_map)
    row_counts_dense = np.apply_along_axis(lambda x: (x == Sparsity.DENSE.value).sum(), 1, sparsity_map)
    sparse_row_offsets = np.concatenate(([0],  np.cumsum(row_counts)))
    dense_row_offsets = np.concatenate(([0],  np.cumsum(row_counts_dense)))

    def store(T, ti, tj, F, fi, fj, prune = None):
        if prune is None or prune == Sparsity.DENSE.value:
            T[m * ti : m * (ti + 1), k * tj : k * (tj + 1)] = \
                F[m * fi : m * (fi + 1), k * fj : k * (fj + 1)]
        elif prune == Sparsity.NV_24.value:
            T[m * ti : m * (ti + 1), k * tj : k * (tj + 1)] = \
                prune_2_4(F[m * fi : m * (fi + 1), k * fj : k * (fj + 1)])
        elif prune == Sparsity.EMPTY.value:
            T[m * ti : m * (ti + 1), k * tj : k * (tj + 1)] = 0
        else:
            raise Exception("Unknown pruning method.")

    # Prune A per tile
    for i in range(rows):
        for j in range(cols): store(A, i, j, A, i, j, prune = sparsity_map[i][j])

    # Decompose into sparse and dense buffers
    total_sparse = np.sum(row_counts)
    total_dense = np.sum(row_counts_dense)
    s = torch.zeros((m, k * total_sparse))
    d = torch.zeros((m, k * total_dense))
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

def generate_matrices(shape, sparse, empty, keep = False):
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


def check_correctness(shape):
    n_sparse = random.randint(shape.tiles_row, shape.tiles_row * shape.tiles_col)
    n_empty = 0
    normal_args, tiled_args = generate_matrices(shape, n_sparse, n_empty, keep=True)

    z = tiled(*tiled_args)
    z_ref = torch.mm(*normal_args)

    # Correctness
    print(f'z_ref - z: {z_ref-z}')
    if torch.allclose(z, z_ref, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
        success = True
    else:
        print("❌ Triton and Torch differ")
        print(f'relative error: {torch.norm(z_ref - z)/torch.norm(z_ref)}')
        diff = [row for row in range(shape.M) if not torch.allclose(z_ref[row],  z[row], atol=1e-2, rtol=1e-2)]
        print(z)
        print(len(diff))
        success = True

    print(f'maximum error: {torch.max(z_ref-z)}, index: {torch.argmax(z_ref-z)}')
    return success

if __name__ == "__main__":
    assert(check_correctness(SHAPE))

    print("Shape:", SHAPE.M, SHAPE.N, SHAPE.K, SHAPE.m, SHAPE.n, SHAPE.k)
    # %%
    # Benchmark
    # ---------
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["sparsity"],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            x_vals=list(np.linspace(SHAPE.tiles_row, SHAPE.tiles_col * SHAPE.tiles_row, 8, dtype=np.int32)),
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["sparse", "tiled", "dense"],  # Label name for the lines
            line_names=["Triton Sparse", "Triton Tiled", "Triton Dense"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="varying_sparsity_global_full",
            args={}
        ))

    @triton.testing.perf_report(configs)
    def benchmark(sparsity, provider):
        if provider == "tiled":
            tiled_args = generate_matrices(SHAPE, sparsity, empty=0)
        else:
            x = torch.randn((SHAPE.M, SHAPE.K), device='cuda', dtype=torch.float16)
            y = torch.randn((SHAPE.K, SHAPE.N), device='cuda', dtype=torch.float16)
            normal_args = (x, y)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "dense":
            ms, min_ms, max_ms = testing.do_bench(lambda: matmul(*normal_args), quantiles=quantiles)
        if provider == "tiled":
            ms, min_ms, max_ms = testing.do_bench(lambda: tiled(*tiled_args), quantiles=quantiles)
        if provider == "sparse":
            x_compressed = CompressedSparse.NV24(*compress_dense_to_sparse( prune_2_4(normal_args[0]) ))
            ms, min_ms, max_ms = testing.do_bench(lambda: matmul(x_compressed, normal_args[1]), quantiles=quantiles)

        perf = lambda ms: 2 * SHAPE.M * SHAPE.N * SHAPE.K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True)