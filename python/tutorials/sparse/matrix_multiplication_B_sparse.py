import os

import torch

import triton as triton
import triton.language as tl
from prune import prune_2_4
from compress import compress_dense_to_sparse

from triton.sparsity.compressed_sparse import CompressedSparse

import argparse

def parse_args(parser):
    parser.add_argument('--bs', help="Batch size", default=16, type=int)

    return parser.parse_args()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        # Small batch sizes
         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                     num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                     num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                     num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                     num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                     num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5,
                     num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5,
                     num_warps=2)
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
                sizes.append((bs, inner_dims[0], inner_dims[1]))

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
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
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
                               offsets=(0, pid_m*BLOCK_SIZE_M),
                               block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
                               order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(N, K),
                               strides=(stride_bk, stride_bn),
                               offsets=(pid_n*BLOCK_SIZE_N, 0),
                               block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
                               order=(1,0))

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, boundary_check=[1], padding_option="zero")
        b = tl.load(b_ptrs, boundary_check=[1], padding_option="zero")
        
        accumulator = tl.dot(b, a, accumulator)

        a_ptrs = tl.advance(a_ptrs, [BLOCK_SIZE_K, 0])
        b_ptrs = tl.advance(b_ptrs, [0, BLOCK_SIZE_K])

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_ptrs = tl.make_block_ptr(c_ptr,
                               shape=(M, N),
                               strides=(stride_cm, stride_cn),
                               offsets=(pid_m*BLOCK_SIZE_M, pid_n*BLOCK_SIZE_N),
                               block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                               order=(1,0))
    tl.store(c_ptrs, c.trans(), boundary_check=(1,0))


def matmul(a, b):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(1), a.stride(0),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def pT(x) : return x.transpose(0, 1).contiguous()

def check_correctness():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

    b_pruned = prune_2_4(pT(b))

    torch_output = torch.matmul(a, pT(b_pruned))
    print(f'ref: {torch_output}')

    b_data, b_metadata = compress_dense_to_sparse(b_pruned)
    b_compressed = CompressedSparse.NV24(b_data, b_metadata)

    triton_output = matmul(a, b_compressed)

    print(f"triton output: {triton_output}")
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
            x_vals = get_tensor_sizes((args.bs,), "init_results"),
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=[ref_lib.lower(), "tritons"],  # Label name for the lines
            line_names=[ref_lib, "Triton Sparse"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-"),  ("orange", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-B-fp16",
            args={}
        ))
    
    print(f'tensor sizes: {get_tensor_sizes((args.bs,), "init_results")}')

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

        b_data, b_metadata = compress_dense_to_sparse(pT(b))
        b_compressed = CompressedSparse.NV24(b_data, b_metadata)

        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'tritons':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b_compressed), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    main(args)