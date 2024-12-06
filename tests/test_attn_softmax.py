import torch
import numpy as np
from typing import Tuple, Union
import pytest
import torch.nn.functional as F
import math

from zhilight.internals_ import functions


class ScaleMaskBiasSoftmax(torch.nn.Module):
    def __init__(
        self,
        scale: float,
        pos_bias_type: str = "relative",
        dtype=torch.half,
    ):
        super().__init__()
        self.scale = scale
        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        score: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score = score * self.scale
        batch_size = 1 if score.ndim == 3 else score.shape[0]
        len_q = score.shape[-2]
        len_k = score.shape[-1]
        print(score.shape, attention_mask.shape, position_bias.shape)
        if self.pos_bias_type == "relative":
            score = score + position_bias
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.view(1, len_q, len_k)
        else:
            attention_mask = attention_mask.view(batch_size, 1, len_q, len_k)
        score = torch.masked_fill(
            score,
            attention_mask == 0,
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )
        score = self.softmax(score)
        score = torch.masked_fill(
            score,
            attention_mask == 0,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )
        return score


"""
 const core::Tensor &attn_score,       // (batch, num_heads, len_q, len_buf)
    const core::Tensor &mask,             // (batch, len_q, len_buf)
    const core::Tensor &position_bias     // if relative (batch, num_head, len_q, len_buf) else if core::Tensor()
"""


@pytest.mark.parametrize(
    "SIZE", [(1, 2, 4, 2), (2, 2, 4, 2), (4, 2, 4, 2)]
)  # batch, num_heads, len_q, len_buf
@pytest.mark.parametrize(
    "SCALE",
    [
        1,
    ],
)
@pytest.mark.parametrize("BIAS_TYPE", ["relative"])
def test_attn_softmax(SIZE, SCALE, BIAS_TYPE):
    rtol, atol = (1e-3, 3e-4)

    mask = torch.randint(
        0, 2, (SIZE[0], SIZE[2], SIZE[3]), dtype=torch.int8, device="cuda"
    ).squeeze(0)
    score = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    position_bias = torch.randn(SIZE, dtype=torch.half, device="cuda").squeeze(0)
    ff_pt = ScaleMaskBiasSoftmax(SCALE, BIAS_TYPE, dtype=torch.half).cuda()

    # if BIAS_TYPE:
    out = functions.attn_softmax(
        SCALE, score.cpu().numpy(), mask.cpu().numpy(), position_bias.cpu().numpy()
    )

    out_pt = ff_pt.forward(score, mask, position_bias)

    print(out)
    print(out_pt.to(torch.float))
    assert torch.allclose(
        torch.from_numpy(out).to(torch.half), out_pt.cpu(), rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    test_attn_softmax((2, 2, 4, 2), 2, "relative")
