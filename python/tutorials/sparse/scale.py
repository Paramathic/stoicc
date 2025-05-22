import torch
import triton
import triton.language as tl

from prune import prune_tensor, prune_2_4
from compress import compress_dense_to_sparse
from triton.sparsity.compressed_sparse import CompressedSparse

torch.set_printoptions(profile="full")

dim = 64

@triton.jit
def scale_kernel(a_ptr,
                 scale,
                 M: tl.constexpr, N: tl.constexpr, # Size of the tensor
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr # Size of the tiles
                 ):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Create the pointers for the offsets
    a_ptrs = tl.make_block_ptr(
        a_ptr,
        shape=(M,N),
        strides=(M,1),
        offsets=(pid_y*BLOCK_M, pid_x*BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0)
    )

    # Load the elements and scale them
    _a = tl.load(a_ptrs)
    _a = scale * _a

    # Store the results back into the tensor
    tl.store(a_ptrs, _a)

def scale (x, s):
    dim1, dim2 = min(x.shape[0], 128), min(x.shape[1], 128)
    grid = lambda meta : (x.shape[0]//dim1, x.shape[1]//dim2,)

    scale_kernel[grid](x, s, x.shape[0], x.shape[1], dim1, dim2)

a = torch.randn((dim, dim), dtype=torch.float16, device=torch.device('cuda')).contiguous()
a = prune_2_4(a)
print(f'a pruned: {a}')

a_data, a_meta = compress_dense_to_sparse(a)
a_compressed = CompressedSparse.NV24(a_data, a_meta)

print('scaling the tensor a  by 2...')
scale(a_compressed, 2)
print(f'a_compressed: {a_compressed}')
print(f'a_compressed metadata: {a_compressed.metadata}')
