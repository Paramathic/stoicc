import torch, triton
import triton.language as tl

from prune import prune_tensor, prune_2_4
from marlin_compress import sparse_semi_structured_from_dense_cutlass
from triton.sparsity.compressed_sparse import CompressedSparse

M, N, K = 512, 512, 512

configs = [triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64},
                         num_warps=4, num_stages=2)]

@triton.autotune(configs=configs,
                 key=['a_ptr', 'M', 'N', 'K'])
@triton.jit
def dot_kernel(
        d_ptr, a_ptr, b_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_row = tl.program_id(0) # The row block id
    pid_col = tl.program_id(1) # The column block id

    a_ptrs = tl.make_block_ptr(a_ptr,
                               shape=(M,K),
                               strides=(K,1),
                               offsets=(pid_row*BLOCK_M, 0),
                               block_shape=(BLOCK_M, BLOCK_K),
                               order=(1,0))
    b_ptrs = tl.make_block_ptr(b_ptr,
                               shape=(K,N),
                               strides=(N,1),
                               offsets=(0, pid_col*BLOCK_N),
                               block_shape=(BLOCK_K, BLOCK_N),
                               order=(1,0))
    d_ptrs = tl.make_block_ptr(d_ptr,
                               shape=(M,N),
                               strides=(N,1),
                               offsets=(pid_row*BLOCK_M, pid_col*BLOCK_N),
                               block_shape=(BLOCK_M, BLOCK_N),
                               order=(1,0))

    num_tiles = tl.cdiv(K, BLOCK_K) # Number of tiles to iterate through in the K dimension
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32) # Accumulator
    # Main Loop
    for i in range(num_tiles):
        a = tl.load(a_ptrs, boundary_check=(0,1), padding_option='zero')
        b = tl.load(b_ptrs, boundary_check=(0,1), padding_option='zero')

        acc = tl.dot(a, b, acc)

        # Advance a and b pointers to the next tiles
        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_K])
        b_ptrs = tl.advance(b_ptrs, [BLOCK_K, 0])

    acc = acc.to(tl.float16)

    tl.store(d_ptrs, acc,  boundary_check=(0,1))

def triton_dot(a, b):
    d = torch.zeros(a.shape[0], b.shape[1]).cuda().half()
    grid = lambda META: (triton.cdiv(a.shape[0], META['BLOCK_M']),
                         triton.cdiv(b.shape[1], META['BLOCK_N'],))
    dot_kernel[grid](d, a, b,
                     M, N, K)

    return d

x = torch.randn(M, K).cuda().half()
y = torch.randn(K, N).cuda().half()

x = prune_2_4(x)
print(f'a pruned: {x}')

x_data, x_metadata = sparse_semi_structured_from_dense_cutlass(x)
x_compressed = CompressedSparse.NV24(x_data, x_metadata)

z = triton_dot(x_compressed, y)
z_ref = torch.mm(x, y)

print(f'z_ref - z: {z_ref-z}')
if torch.allclose(z, z_ref, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match")
    print(f'maximum error: {torch.max(z_ref-z)}, index: {torch.argmax(z_ref-z)}')
else:
    print("❌ Triton and Torch differ")
    print(f'relative error: {torch.norm(z_ref - z)/torch.norm(z_ref)}')
    print(f'maximum error: {torch.max(z_ref-z)}, index: {torch.argmax(z_ref-z)}')