import pytest
import torch
import torch.nn.functional as F
from zhilight.internals_ import functions


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_index_along_dim(SIZE):
    rtol, atol = (1e-3, 3e-4)

    index_shape = list(SIZE[:-1])
    index_shape[-1] = 2
    input = torch.randn(SIZE, dtype=torch.half, device="cuda")  # .squeeze(0)
    index = torch.randint(
        0, SIZE[2], index_shape, dtype=torch.int32, device="cuda"
    )  # .squeeze(0)
    out = functions.index_along_dim(input, 2, index)
    out_pt = torch.take_along_dim(input, index[:, :, :, None].to(torch.long), 2)
    print(input)
    print(index)
    print(out)
    print(out_pt)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_index_along_dim((2, 2, 4, 2))
