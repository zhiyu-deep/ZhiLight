import pytest
import torch
import torch.nn.functional as F
from zhilight.internals_ import functions


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_softmax(SIZE):
    rtol, atol = (1e-3, 3e-4)

    score = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    out_pt = F.softmax(score, dim=-1)
    out = functions.softmax(score, 1.0)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_softmax((2, 2, 4, 2))
