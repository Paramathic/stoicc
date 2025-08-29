# Sample Sparse Executor
#  This file provides sample code for a sparse executor that
#  takes a reordered matrix and schedule created by an inspector
#  and performs a SpMM operation using the 2:4 backend.

import torch, triton, time
import sys
from pathlib import Path
import triton.language as tl

from triton import testing

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mixed_tile_inspector import * # Import inspector code

# Parameters
# Optimal tile sizes/other parameters should be selected by autotuning a 2:4/dense kernel of the same size.
SHAPE = MMAShape(4096, 16, 4096, # Matrix dimensions M, N, K
                 128, 16, 64,     # Tile size along each matrix dimension
                 num_warps = 8, num_stages = 3, group_size = 8, sparseA = True)

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
def split_stream_kernel(
        # Pointers to matrices
        a_ptr, a_col, a_row,
        b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K, RESHAPED_K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,    #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """
    This kernel should be launched on both the sparse and dense data independently. The launch of
    the two kernels allows enables the parallel computation of dense and sparse output data.
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

    a_row += pid_row
    row_l = tl.load(a_row)
    row_r = tl.load(a_row + 1)


    a_col += pid_row * tl.cdiv(K, BLOCK_SIZE_K)
    first_b = tl.load(a_col)
    a_col += 1

    a_ptrs = tl.make_block_ptr(a_ptr,
                                shape=(BLOCK_SIZE_M,RESHAPED_K), strides=(stride_am, stride_ak), offsets=(0, row_l*BLOCK_SIZE_K),
                                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(K,N), strides=(stride_bk, stride_bn), offsets=(first_b*BLOCK_SIZE_K, pid_col*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1,0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Sparse Loop
    for k in range(row_r - row_l):
        accumulator = tl.dot(
            tl.load(a_ptrs, boundary_check=[1], padding_option="zero"),
            tl.load(b_ptrs, boundary_check=[1], padding_option="zero"),
            accumulator
        )

        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_SIZE_K])
        b_ptrs = tl.advance(b_ptrs, [tl.load(a_col) * BLOCK_SIZE_K, 0])
        a_col += 1

    c = accumulator.to(tl.float16)

    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M,N), strides=(stride_cm,stride_cn), offsets=(pid_row*BLOCK_SIZE_M, pid_col*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1,0))

    tl.store(c_ptrs, c, boundary_check=(1,0))

def execute_split_stream_tiled(a_sparse, a_dense, b, sparse_col, dense_col, a_sparse_row, a_dense_row):
    M, N, K = SHAPE.M, SHAPE.N, SHAPE.K
    SK, DK = a_sparse.shape[1], a_dense.shape[1]
    
    # Create a stream for dense and a stream for sparse
    sparse_stream = torch.cuda.Stream(device=b.device)
    dense_stream = torch.cuda.Stream(device=b.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    with torch.cuda.stream(sparse_stream):
        c_sparse = torch.zeros((M, N), device=b.device, dtype=torch.float16)
        split_stream_kernel[grid](
            a_sparse, sparse_col, a_sparse_row, b, c_sparse,
            M, N, K, SK,
            a_sparse.stride(0), a_sparse.stride(1),
            b.stride(0), b.stride(1),
            c_sparse.stride(0), c_sparse.stride(1)
        )

    with torch.cuda.stream(dense_stream):
        c_dense = torch.zeros((M, N), device=b.device, dtype=torch.float16)
        split_stream_kernel[grid](
            a_dense, dense_col, a_dense_row, b, c_dense,
            M, N, K, DK,
            a_dense.stride(0), a_dense.stride(1),
            b.stride(0), b.stride(1),
            c_dense.stride(0), c_dense.stride(1),
        )    

    consumer = torch.cuda.current_stream(device=b.device)
    consumer.wait_stream(sparse_stream)
    consumer.wait_stream(dense_stream)

    return c_dense + c_sparse

def check_correctness(shape):
    # n_sparse = random.randint(shape.tiles_row, shape.tiles_row * shape.tiles_col)
    n_sparse = random.randint(shape.tiles_row* shape.tiles_col, shape.tiles_row * shape.tiles_col)
    n_empty = 0

    # Inspection phase assigns sparsity types to tiles and performs reordering/scheduling
    normal_args, tiled_args = inspect_tiled(shape, n_sparse, n_empty, keep=True, separate_col_order=True)

    # Exceution phase performs the matmul
    # z = execute_tiled(*tiled_args)
    z = execute_split_stream_tiled(*tiled_args)
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
            # Inspection Phase
            tiled_args = inspect_tiled(SHAPE, sparsity, empty=0, separate_col_order=True)
        else:
            x = torch.randn((SHAPE.M, SHAPE.K), device='cuda', dtype=torch.float16)
            y = torch.randn((SHAPE.K, SHAPE.N), device='cuda', dtype=torch.float16)
            normal_args = (x, y)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "dense":
            ms, min_ms, max_ms = testing.do_bench(lambda: matmul(*normal_args), quantiles=quantiles)
        if provider == "tiled":
            # Execution Phase
            ms, min_ms, max_ms = testing.do_bench(lambda: execute_split_stream_tiled(*tiled_args), quantiles=quantiles)
        if provider == "sparse":
            x_compressed = CompressedSparse.NV24(*compress_dense_to_sparse( prune_2_4(normal_args[0]) ))
            ms, min_ms, max_ms = testing.do_bench(lambda: matmul(x_compressed, normal_args[1]), quantiles=quantiles)

        perf = lambda ms: 2 * SHAPE.M * SHAPE.N * SHAPE.K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True)