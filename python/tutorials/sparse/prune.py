import torch

# torch.set_printoptions(profile="full")

def gen_mask(device: torch.device,
             size: tuple[int, int],
             n: int,
             m: int):
    """
    Generate a tensor of shape [1,m] where at most
    n of the elements are non-zero
    :param device: The device on which to generate the tensor
    :param n: The maximum nnz
    :param m: The length of the tensor in the second dimension
    :return: The tensor that is generated
    """
    if n > m:
        raise ValueError("n cannot be greater than m.")

    # Create (m - n) zeros and n ones
    zeros = torch.zeros(m - n, dtype=torch.int32, device=device)
    ones = torch.ones(n, dtype=torch.int32, device=device)

    # Concatenate zeros and ones
    tensor = torch.cat((zeros, ones))

    # Shuffle the tensor to distribute zeros and ones randomly
    indices = torch.randperm(m, device=device)

    ret = tensor[indices].repeat(size)
    return ret




def prune_tensor(x: torch.Tensor,
                 n: int=2,
                 m: int=4):
    """
    Prune the tensor m:n
    :param x: The tensor to prune
    :param n: The n value in n:m sparsity
    :param m: The m value in n:m sparsity
    :return:
    """
    device = x.device
    assert len(x.shape) == 2, "The tensor must be 2-dimensional"
    assert x.shape[1] % m == 0, ("The tensor's second dimension must be divisible"
                                 "by the values of m")

    print(f'generating mask')

    mask = gen_mask(device,
                    size=(x.shape[0], x.shape[1]//m),
                    n=n,
                    m=m)
    print(f'mask generated')
    x *= mask

    return x


def prune_2_4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Prune a tensor in a 2:4 fashion. For each contiguous block of 4 elements
    along the last dimension, the two smallest values are replaced with 0.

    For example, if a block is [1, 5, 3, 2], then the two smallest values (1 and 2)
    will be pruned, and the resulting block will be [0, 5, 3, 0].

    Args:
        tensor (torch.Tensor): Input tensor. The size of its last dimension must be divisible by 4.

    Returns:
        torch.Tensor: A new tensor with the same shape as the input, where in each group of 4
                      the two smallest values have been set to 0.

    Raises:
        ValueError: If the size of the last dimension is not divisible by 4.
    """
    # Check that the last dimension is divisible by 4.
    if tensor.size(-1) % 4 != 0:
        raise ValueError("The size of the last dimension must be divisible by 4.")

    # Reshape the tensor so that the last dimension groups elements in sets of 4.
    # For example, if tensor.shape is (N, M) with M divisible by 4, we reshape to (N, M//4, 4).
    new_shape = tensor.shape[:-1] + (tensor.shape[-1] // 4, 4)
    tensor_grouped = tensor.view(new_shape)

    # For each group of 4 elements, find the indices of the 2 smallest values.
    # torch.topk with largest=False returns the smallest values.
    _, smallest_indices = torch.topk(tensor_grouped, k=2, dim=-1, largest=False)

    # Create a boolean mask of the same shape as tensor_grouped.
    mask = torch.zeros_like(tensor_grouped, dtype=torch.bool)
    # Scatter True into the mask at the positions of the two smallest values.
    mask.scatter_(-1, smallest_indices, True)

    # Replace the two smallest values with 0.
    pruned_grouped = tensor_grouped.masked_fill(mask, 0)

    # Reshape back to the original shape.
    pruned_tensor = pruned_grouped.view(tensor.shape)

    return pruned_tensor