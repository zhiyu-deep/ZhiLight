from typing import List, Optional

from zhilight.server.openai.basic.sequence import (PromptLogprobs, SampleLogprobs, RequestMetrics)
from zhilight.server.openai.lora.request import LoRARequest


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids_num: int,
        cumulative_logprob: float,
        logprobs: Optional[SampleLogprobs],
        finish_reason: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids_num = token_ids_num
        self.token_ids = [] # temp compatible
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.lora_request = lora_request

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids_num={self.token_ids_num}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids_num: int,
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = [] # temp compatible
        self.prompt_token_ids_num = prompt_token_ids_num
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids_num={self.prompt_token_ids_num}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, "
                f"lora_request={self.lora_request})")
