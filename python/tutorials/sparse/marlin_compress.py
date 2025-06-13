import torch
# Taken from the Marlin-Sparse Codebase


# This is PyTorch implementation of main part of reorder_meta()
# function, from tools/util/include/cutlass/util/host_reorder.h file
# of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
# GEMM decides upon layout of this matrix, and at the moment for the
# sparse GEMM executed on tensor cores, this is layout described by
# ColumnMajorInterleaved<2> data structure, in
# include/cutlass/layout/matrix.h of CUTLASS source tree.  The
# reordering of meta matrix into meta_reordered matrix calculated
# according to these segments of CUTLASS code is re-implemented here.
# Note that this calculation produces offsets for scattering metadata
# matrix elements into reordered metadata matrix elements (or,
# equivalently, for gathering reordered metadata matrix element back
# into metadata matrix elements).
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group_x = 64
    group_y = 32 if meta_dtype.itemsize == 2 else 16

    dst_rows = (
            dst_rows // group_x * group_x
            + (dst_rows % 2) * 2
            + (dst_rows % 8) // 4
            + ((dst_rows % group_y) % 4) // 2 * 32
            + ((dst_rows % group_x) // 8) * 4
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)

def _calculate_paramath_metadata_reordering_offsets(m, meta_ncols, device):
    ''' This is for a sparse matrix A that is 2:4 and has shape m16n8k16
    In this case, there is a 16 bit metadata element per row, meaning
    that each 16x16 tile has a 16x1xINT16 tiled metadata.

    Each thread in a group of four consecutive threads in a warp provide 2 of these
    int16 values in a 32bit register. A matrix of the following shape

    0_lo
    .
    .
    7_lo
    0_hi
    .
    .
    7_hi

    must be mapped to a 8x2xint16 shape
    0_hi, 0_lo
    .
    .
    7_hi, 7_lo

    i.e. if A is a 8x1xint16 metadata subtile, A_lo is providing the lowest 16 bits and
    A_hi is providing the highest 16 bits to the same compressed data tile, then we have the following
    transformation in this function

    ---------------                 -----------------------------
    | A_lo | B_lo |                 | A_lo | A_hi | B_lo | B_hi |
    ---------------                 -----------------------------
    | A_hi | B_hi |                 | C_lo | C_hi | D_lo | D_hi |
    ---------------        ==>      -----------------------------
    | C_lo | D_lo |
    ---------------
    | C_hi | D_hi |
    ---------------

    '''
    assert meta_ncols%4 == 0, f"The number of metadata cols must be at least 4, current number: {meta_ncols}"
    offsets = torch.zeros(m*meta_ncols, device=device, dtype=torch.int64)
    for i in range(m*meta_ncols):
        row = i // meta_ncols
        col = i % meta_ncols

        new_stride = meta_ncols * 16
        groupId = row % 8


        dst_row = row//16
        dst_col = ((col//4) * 64) + \
                  (groupId * 8) + \
                  ((row//8) % 2)+((col%4) * 2)

        offsets[i] = dst_row*new_stride + dst_col
    return offsets


def sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float, torch.int32]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1 # expression will be 0100
    expr1 = ~m0 & m1 # will end in 01
    expr2 = ~m0 & ~m1 # expression will be 1110
    bit0 = expr1 # right most bit is only 1 when the index=1 is nz and index=0 is not
    bit1 = expr2 # 2nd right most bit is only 1 index=2 is the smallest-indexed nz
    bit2 = expr0 | expr2 | m3 # bit 2 (2nd most left bit is true in these three cases)
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        # Using the indices we have, create the sparse matrix (the values)
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2) # metadata in 4 bit chunks (this is how we leave it when we store right now!)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        # Do the shifts so that the order of metadata is respected in the load later
        # i.e. this part does:
        # byte0 | byte1 | byte2 | byte3     ==>     byte3 | byte2 | byte1 | byte0
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
                | (meta_n[:, :, 4] << 16)
                | (meta_n[:, :, 5] << 20)
                | (meta_n[:, :, 6] << 24)
                | (meta_n[:, :, 7] << 28)
        )

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    # meta_offsets = _calculate_meta_reordering_scatter_offsets(
    #     m, meta_ncols, meta_dtype, device
    # )
    meta_offsets = _calculate_paramath_metadata_reordering_offsets(m, meta_ncols, device)
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return sparse, meta_reordered.view(-1, meta_ncols*2)
    # return sparse, meta