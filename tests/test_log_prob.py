import torch
import pytest
from torch.nn import functional as F
from zhilight.internals_ import functions


@pytest.mark.parametrize("SIZE", [(4, 4)])
@pytest.mark.parametrize("EXT_SPLIT", [2])
def test_log_prob(SIZE, EXT_SPLIT):
    rtol, atol = (1e-3, 3e-4)

    INPUT_SIZE = (SIZE[0], EXT_SPLIT)
    INPUT_EXT_SIZE = (SIZE[0], SIZE[1] - EXT_SPLIT)

    input = torch.normal(0, 1, size=SIZE)
    input_list = input[:, :EXT_SPLIT].reshape(1, -1).squeeze().tolist()
    input_ext_list = input[:, EXT_SPLIT:].reshape(1, -1).squeeze().tolist()

    labels = torch.randint(0, SIZE[1], (SIZE[0],))
    label_list = labels.tolist()

    # log_prob() returns negitave log-likelihood loss tuple: (sum reduced, unreduced)
    nll_loss, neg_lsf = functions.log_prob(
        input_list, INPUT_SIZE, input_ext_list, INPUT_EXT_SIZE, label_list
    )

    lsf = F.log_softmax(input, dim=-1)
    neg_lsf_ref = F.nll_loss(lsf, labels, reduction="none")
    nll_loss_ref = F.nll_loss(lsf, labels, reduction="sum")

    assert torch.allclose(torch.tensor(neg_lsf), neg_lsf_ref, rtol=rtol, atol=atol)
    assert torch.allclose(torch.tensor(nll_loss), nll_loss_ref, rtol=rtol, atol=atol)

    def logprobs_from_logits(logits, labels, dim=-1):
        """
        See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
        """
        # add support for half
        logp = F.log_softmax(logits.float(), dim=dim)
        logpy = torch.gather(logp, dim, labels.unsqueeze(dim)).squeeze(-1)
        logpy = logpy.type(logits.dtype)
        return logpy

    log_probs = logprobs_from_logits(input, labels)

    assert torch.allclose(-torch.tensor(neg_lsf), log_probs, rtol=rtol, atol=atol)
