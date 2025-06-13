import torch, triton
import triton.language as tl

M, N, K = 512, 512, 512
M_BLOCK = 32
N_BLOCK = 32
K_BLOCK = 64
configs = [triton.Config({}, num_warps=4, num_stages=3)]

@triton.autotune(configs=configs, key=['a_ptr'])
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
        # a = tl.load(a_ptrs)
        # b = tl.load(b_ptrs)

        acc = tl.dot(a, b, acc)

        # Advance a and b pointers to the next tiles
        a_ptrs = tl.advance(a_ptrs, [0, BLOCK_K])
        b_ptrs = tl.advance(b_ptrs, [BLOCK_K, 0])

    acc = acc.to(tl.float16)

    tl.store(d_ptrs, acc, boundary_check=(0,1))
    # tl.store(d_ptrs, acc)

def triton_dot(a, b):
    d = torch.zeros(a.shape[0], b.shape[1]).cuda().half()
    grid = lambda meta: (triton.cdiv(a.shape[0],M_BLOCK), triton.cdiv(b.shape[1],N_BLOCK),)
    dot_kernel[grid](d, a, b,
                     M, N, K,
                     M_BLOCK, N_BLOCK, K_BLOCK)

    return d

x = torch.randn(M, K).cuda().half()
y = torch.randn(K, N).cuda().half()

assert x.is_contiguous() and y.is_contiguous()

z = triton_dot(x, y)
z_ref = torch.mm(x, y)

print(f'z_ref - z: {z_ref-z}')
if torch.allclose(z, z_ref, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
    print(f'maximum error: {torch.max(z_ref-z)}')
else:
    print("❌ Triton and Torch differ")
    print(f'relative error: {torch.norm(z_ref - z)/torch.norm(z_ref)}')
    print(f'maximum error: {torch.max(z_ref-z)}')