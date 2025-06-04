'''
There is no difference between this and the third tutorial for
dense matrix multiplication, other than the preprocessing that happens before
the matmul kernel is launched.

The following lines
    1. prune a matrix 2:4 using magnitude pruning
    2. compress the data to its values and metadata arrays in PyTorch
    (User can replace both of these steps)
    3. Create a compressed values matrix that Triton will recognise as 2:4
    using the data and metadata arrays from the previous step

```
a = prune_2_4(a)
a_data, a_metadata = compress_dense_to_sparse(a)
a_compressed = CompressedSparse.NV24(a_data, a_metadata)
```
If the compressed sparse matrix is passed to the kernel, it will be recognised,
any operations that use the kernel will be interpreted as sparse operations,
and the sparse code will be generated for them.
'''

import torch

import triton as triton
import triton.language as tl
from prune import prune_2_4
from compress import compress_dense_to_sparse
from triton.sparsity.compressed_sparse import CompressedSparse

import argparse

def parse_args(parser):
    parser.add_argument('--path', help="the path for saving the results", required=True, type=str)
    parser.add_argument('--plot_name', default="matmul-performance-fp16",
                        help="Name of the plot to save", type=str)
    parser.add_argument('--bs', help="Batch size", default=16, type=int)

    return parser.parse_args()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def get_cuda_autotune_config():
    return [
        # 16 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=16),
        # 8 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        # 4 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),

        # Small N
        # 16 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=16),
        # 8 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        # 4 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=4),
    ]

def get_split_k_autotune_config():
    return [
        # 16 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=16, pre_hook=init_to_zero('c_ptr')),
        # 8 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        # 4 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),

        # Small N
        # 16 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=16, pre_hook=init_to_zero('c_ptr')),
        # 8 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=8, pre_hook=init_to_zero('c_ptr')),
        # 4 warps
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
                      num_stages=3, num_warps=4, pre_hook=init_to_zero('c_ptr')),
    ]

def get_init_result_sizes():
    return [ # Initial results OPT model sizes
        (21504, 7168), (7168, 7168), (28672, 7168), (7168, 28672),
        (27648, 9216), (9216, 9216), (36864, 9216), (9216, 36864),
        (36864, 12288), (12288, 12288), (12288, 49152), (49152, 12288)
    ]

def get_tensor_sizes(batch_sizes=(512,), method="LLM"):
    weight_dims = []
    sizes = []

    if method == "init_results":
        weight_dims.extend(get_init_result_sizes())
        for bs in batch_sizes:
            for inner_dims in weight_dims:
                sizes.append((inner_dims[0], bs, inner_dims[1]))

    return sizes


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        raise Exception("Only CUDA is supported for 2:4 mamtul")

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
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
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
    configs=get_split_k_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel_sparse_split_k(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # The K dimension is split between SPLIT_K CTAs and each CTA
    # does the computation for K/(BLOCK_SIZE_K*SPLIT_K) tiles.
    # The tiles that each CTA works on are strided throughout the matrix
    # i.e. the work along K is split like this:
    #
    # -------------------------------------
    # | CTA 0 | CTA 1 | CTA 0 | CTA 1 | ...
    # -------------------------------------
    #
    a_ptrs = tl.make_block_ptr(a_ptr,
                               shape=(M, K),
                               strides=(stride_am, stride_ak),
                               offsets=(pid_m*BLOCK_SIZE_M, pid_k*BLOCK_SIZE_K),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                               order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(K, N),
                               strides=(stride_bk, stride_bn),
                               offsets=(pid_k*BLOCK_SIZE_K, pid_n*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                               order=(1,0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_tiles = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    for k in range(num_tiles):
        a = tl.load(a_ptrs, boundary_check=(0,1), padding_option="zero")
        b = tl.load(b_ptrs, boundary_check=(0,1), padding_option="zero")

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_SIZE_K*SPLIT_K])
        b_ptrs = tl.advance(b_ptrs, [BLOCK_SIZE_K*SPLIT_K, 0])

    accumulator = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    # Pointers for the output tile
    _m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    _n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (_m[:, None]*stride_cm + _n[None,:]*stride_cn)

    # Reduce to the output tile over the different CTAs
    tl.atomic_add(c_ptrs, accumulator, sem="relaxed")

def matmul_split_k(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocate zeros since we will be adding to it
    # Speed this up
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
                         META['SPLIT_K'])
    matmul_kernel_sparse_split_k[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

def check_correctness ():
    # %%
    # Unit Test
    # ---------
    #
    # We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    a = prune_2_4(a)
    a_data, a_metadata = compress_dense_to_sparse(a)
    a_compressed = CompressedSparse.NV24(a_data, a_metadata)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a_compressed, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 1e-2
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

def main(args):
    check_correctness()

    ref_lib = 'cuBLAS Dense'
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=get_tensor_sizes((args.bs,), "init_results"),
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=[ref_lib.lower(), "tritons", "splitk"],  # Label name for the lines
            line_names=[ref_lib, "Triton Sparse", "Sparse splitk"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("Orange", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name=args.plot_name,
            args={}
        ))

    print(f'sizes: {get_tensor_sizes((args.bs,), "init_results")}')

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)

        # Magnitude prune the matrix 2:4
        a = prune_2_4(a)
        # Compress the matrix to its compressed dense values and metadata
        a_data, a_metadata = compress_dense_to_sparse(a)
        # Create a compressed 2:4 matrix type recognised by Triton
        a_compressed = CompressedSparse.NV24(a_data, a_metadata)

        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'tritons':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a_compressed, b), quantiles=quantiles)
        if provider == 'splitk':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_split_k(a_compressed, b), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark.run(print_data=True, save_path=args.path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    main(args)