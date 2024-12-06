import pytest
import torch
import torch.nn.functional as F
from zhilight.internals_ import functions


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_sum(SIZE):
    rtol, atol = (1e-3, 3e-4)

    score = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    out_pt = torch.sum(score, -1, keepdim=True)
    out = functions.sum(score)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_div(SIZE):
    rtol, atol = (1e-3, 3e-4)

    input_a = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    input_b = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    out_pt = input_a / input_b
    out = functions.div(input_a, input_b, 0.0)
    print(out_pt)
    print(out)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_amax(SIZE):
    rtol, atol = (1e-3, 3e-4)

    score = torch.randn(SIZE, dtype=torch.bfloat16, device="cuda").squeeze(0)
    out_pt = torch.amax(score, -1, keepdim=True)
    out = functions.amax(score)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_amin(SIZE):
    rtol, atol = (1e-3, 3e-4)

    score = torch.randn(SIZE, dtype=torch.bfloat16, device="cuda").squeeze(0)
    out_pt = torch.amin(score, -1, keepdim=True)
    out = functions.amin(score)
    print(out)
    print(out_pt)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
def test_add(SIZE):
    rtol, atol = (1e-3, 3e-4)

    score = torch.randint(10, SIZE, dtype=torch.int, device="cuda").squeeze(0)
    out_pt = score + 1
    out = functions.add(score, 1)
    print(out)
    print(out_pt)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_amin((2, 2, 4, 2))
