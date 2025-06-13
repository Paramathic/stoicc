class CompressedSparse:
    @staticmethod
    def get_supported_sparsities():
        return {"NV24"}
    def __init__(self, data, metadata, shape, stride, sparsity_pattern=""):
        self.data = data
        self.metadata = metadata
        self.dtype = "sparse"
        self.element_ty = data.dtype
        self.shape = shape
        self.sparsity_pattern = sparsity_pattern
        self._stride = stride

        assert data.device == metadata.device
        self.device = data.device

    @classmethod
    def NV24(cls, data, metadata):
        assert data.is_contiguous(), "The data provided must be contiguous"
        assert metadata.is_contiguous(), "The metadata provided must be contiguous"
        return cls(data, metadata,
                   shape=(data.shape[0], data.shape[1]*2),
                   stride=(data.stride(0)*2, data.stride(1)),
                   sparsity_pattern="NV24")

    def __repr__(self):
        return self.data.__repr__()

    def is_contiguous(self):
        return self.data.is_contiguous() and self.metadata.is_contiguous()

    def stride(self):
        return self._stride

    def stride(self, n):
        return self._stride[n]


class DenseTensor:
    def __init__(self, handle):
        self.handle = handle

    def data_ptr(self):
        return self.handle

    def __repr__(self):
        return f'DenseTensor[handle={self.handle}]'