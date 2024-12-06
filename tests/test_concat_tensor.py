import torch
import pytest
from torch.nn import functional as F
from zhilight.internals_ import functions


@pytest.mark.parametrize(
    "SIZE",
    [(2, 3, 4), (2, 3, 2)],
)
@pytest.mark.parametrize(
    "DIM",
    [
        -1,
    ],
)
def test_concat_tensor(SIZE, DIM):
    rtol, atol = (1e-3, 3e-4)

    A = torch.normal(0, 1, size=SIZE, dtype=torch.half).cuda()
    B = torch.normal(0, 1, size=SIZE, dtype=torch.half).cuda()

    print(A)
    print(B)
    C = functions.concat_tensor(A.cpu().numpy(), B.cpu().numpy(), DIM)

    C_pt = torch.concat((A, B), dim=DIM)

    print(C)
    print(C_pt)
    assert torch.allclose(torch.tensor(C).half().cuda(), C_pt, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_concat_tensor((2, 3), -2)
