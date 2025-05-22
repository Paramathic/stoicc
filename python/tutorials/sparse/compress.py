# Taken from the Marlin-Sparse Codebase as an example of a user doing 2:4
# matmul compression and passing it in

import torch

def _calculate_meta_reordering_offsets(m, meta_ncols, device):
    '''
    Calculate the reordering offset of the metadata
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


def compress_dense_to_sparse(dense):
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
    meta_reordered = meta.new_empty((m * meta_ncols,))
    meta_offsets = _calculate_meta_reordering_offsets(m, meta_ncols, device)
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return sparse, meta_reordered.view(-1, meta_ncols*2)