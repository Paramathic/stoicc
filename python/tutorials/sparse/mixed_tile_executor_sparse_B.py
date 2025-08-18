# Sample Sparse Executor
#  This file provides sample code for a sparse executor that
#  takes a reordered matrix and schedule created by an inspector
#  and performs a SpMM operation using the 2:4 backend.

import torch, triton
import triton.language as tl

from triton import testing
from mixed_tile_inspector import * # Import inspector code

# Parameters
# Optimal tile sizes/other parameters should be selected by autotuning a 2:4/dense kernel of the same size.
SHAPE = MMAShape(32, 4096, 2048, # Matrix dimensions M, N, K
                 16, 128, 64,     # Tile size along each matrix dimension
                 num_warps = 4, num_stages = 3, group_size = 8, sparseA=False)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': SHAPE.m, 'BLOCK_SIZE_N': SHAPE.n, 'BLOCK_SIZE_K': SHAPE.k, 'GROUP_SIZE_M': SHAPE.group_size},
                      num_stages=SHAPE.num_stages, num_warps=SHAPE.num_warps)
    ]

# Normal Triton Matmul (Baseline)
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
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, boundary_check=(0,1), padding_option="zero")
        b = tl.load(b_ptrs, boundary_check=(0,1), padding_option="zero")

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_SIZE_K])
        b_ptrs = tl.advance(b_ptrs, [BLOCK_SIZE_K, 0])

    c = accumulator.to(tl.float16)

    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M, N),
                               strides=(stride_cm, stride_cn),
                               offsets=(pid_m*BLOCK_SIZE_M, pid_n*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                               order=(1,0))
    tl.store(c_ptrs, c, boundary_check=(1,0))

def matmul(a, b):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )

    return c

# Tiled Kernel
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tiled_kernel(
        # Pointers to matrices
        a_ptr,
        b_sparse_ptr, b_dense_ptr, b_row, b_sparse_row, b_dense_row,
        c_ptr,
        # Matrix dimensions
        M, N, K, SK, DK,
        stride_am, stride_ak,    #
        stride_bsn, stride_bsk,  #
        stride_bdn, stride_bdk,  #
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

    b_sparse_row += pid_col
    b_dense_row += pid_col
    sparse_l = tl.load(b_sparse_row)
    sparse_r = tl.load(b_sparse_row + 1)
    dense_l  = tl.load(b_dense_row)
    dense_r  = tl.load(b_dense_row + 1)

    b_row += pid_col * tl.cdiv(K, BLOCK_SIZE_K)
    first_a = tl.load(b_row)
    b_row += 1

    a_ptrs = tl.make_block_ptr(a_ptr,
                               shape=(M,K), strides=(stride_ak, stride_am), offsets=(first_a*BLOCK_SIZE_K, pid_row*BLOCK_SIZE_M),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M), order=(1,0))
    b_sparse_ptrs = tl.make_block_ptr(b_sparse_ptr,
                                      shape=(BLOCK_SIZE_N,SK), strides=(stride_bsn,stride_bsk), offsets=(0, sparse_l*BLOCK_SIZE_K),
                                      block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), order=(1,0))
    b_dense_ptrs = tl.make_block_ptr(b_dense_ptr,
                                     shape=(BLOCK_SIZE_N,DK), strides=(stride_bdn,stride_bdk), offsets=(0, dense_l*BLOCK_SIZE_K),
                                     block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), order=(1,0))

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    # Sparse Loop
    for k in range(sparse_r - sparse_l):
        accumulator = tl.dot(
            tl.load(b_sparse_ptrs),
            tl.load(a_ptrs),
            accumulator
        )

        a_ptrs = tl.advance(a_ptrs, [tl.load(b_row) * BLOCK_SIZE_K, 0])
        b_sparse_ptrs = tl.advance(b_sparse_ptrs, [0, BLOCK_SIZE_K])
        b_row += 1

    # Dense Loop
    for k in range(dense_r - dense_l):
        accumulator = tl.dot(
            tl.load(b_dense_ptrs),
            tl.load(a_ptrs),
            accumulator
        )

        a_ptrs = tl.advance(a_ptrs, [tl.load(b_row) * BLOCK_SIZE_K, 0])
        b_dense_ptrs = tl.advance(b_dense_ptrs, [0, BLOCK_SIZE_K])
        b_row += 1

    c = accumulator.to(tl.float16)

    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M,N), strides=(stride_cm,stride_cn),
                               offsets=(pid_row*BLOCK_SIZE_M, pid_col*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1,0))

    tl.store(c_ptrs, c.trans())

def execute_tiled(a, b_sparse, b_dense, b_row, b_sparse_row, b_dense_row,):
    M, N, K = SHAPE.M, SHAPE.N, SHAPE.K
    SK, DK = b_sparse.shape[1], b_dense.shape[1]
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    tiled_kernel[grid](
        a,
        b_sparse, b_dense, b_row, b_sparse_row, b_dense_row, #
        c,  #
        M, N, K, SK, DK, #
        a.stride(0), a.stride(1),  #
        b_sparse.stride(0), b_sparse.stride(1),  #
        b_dense.stride(0), b_dense.stride(1),  #
        c.stride(0), c.stride(1),  #
    )

    return c


def check_correctness(shape):
    n_sparse = random.randint(shape.tiles_row, shape.tiles_row * shape.tiles_col)
    n_empty = 0

    # Inspection phase assigns sparsity types to tiles and performs reordering/scheduling
    normal_args, tiled_args = inspect_tiled(shape, n_sparse, n_empty, keep=True)

    # Execution phase performs the matmul
    z = execute_tiled(*tiled_args)
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
        exit()

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
            x_vals=list(np.linspace(SHAPE.tiles_row, SHAPE.tiles_col * SHAPE.tiles_row, 8, dtype=np.int32)),
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # line_vals=["sparse", "tiled", "dense"],  # Label name for the lines
            # line_names=["Triton Sparse", "Triton Tiled", "Triton Dense"],  # Line styles
            line_vals=["tiled", "dense"],  # Label name for the lines
            line_names=["Triton Tiled", "Triton Dense"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="varying_sparsity_global_full",
            args={}
        ))

    @triton.testing.perf_report(configs)
    def benchmark(sparsity, provider):
        if provider == "tiled":
            # Inspection Phase
            tiled_args = inspect_tiled(SHAPE, sparsity, empty=0)
        else:
            x = torch.randn((SHAPE.M, SHAPE.K), device='cuda', dtype=torch.float16)
            y = torch.randn((SHAPE.K, SHAPE.N), device='cuda', dtype=torch.float16)
            normal_args = (x, y)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "dense":
            ms, min_ms, max_ms = testing.do_bench(lambda: matmul(*normal_args), quantiles=quantiles)
        if provider == "tiled":
            # Execution Phase
            ms, min_ms, max_ms = testing.do_bench(lambda: execute_tiled(*tiled_args), quantiles=quantiles)

        perf = lambda ms: 2 * SHAPE.M * SHAPE.N * SHAPE.K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True)