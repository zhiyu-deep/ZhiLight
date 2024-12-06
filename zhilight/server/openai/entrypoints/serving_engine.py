import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional
from zhilight.server.openai.basic.logger import init_logger
from zhilight.server.openai.engine.async_llm_engine import AsyncLLMEngine
from zhilight.server.openai.entrypoints.protocol import (ErrorResponse, LogProbs,
                                              ModelCard, ModelList,
                                              ModelPermission)
from zhilight.server.openai.lora.request import LoRARequest
from zhilight.server.openai.basic.sequence import Logprob

logger = init_logger(__name__)

@dataclass
class LoRA:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 lora_modules=Optional[List[LoRA]]):
        self.engine = engine
        self.served_model = served_model
        if lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                ) for i, lora in enumerate(lora_modules, start=1)
            ]

        self.max_model_len = 0

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            event_loop.create_task(self._post_init())
        else:
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_config = await self.engine.get_engine_config()
        self.max_model_len = engine_config.max_model_len

    # get models list
    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=self.served_model,
                      root=self.served_model,
                      permission=[ModelPermission()])
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=self.served_model,
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is not None:
                token_logprob = step_top_logprobs[token_id].logprob
            else:
                token_logprob = None
            token = step_top_logprobs[token_id].decoded_token
            logprobs.tokens.append(token)
            logprobs.token_logprobs.append(token_logprob)
            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] +
                                            last_token_len)
            last_token_len = len(token)

            if num_output_top_logprobs:
                logprobs.top_logprobs.append({
                    p.decoded_token: p.logprob
                    for i, p in step_top_logprobs.items()
                } if step_top_logprobs else None)
        return logprobs

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str
    # check model exists
    async def _check_model(self, request) -> Optional[ErrorResponse]:
        if request.model == self.served_model:
            return
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)
    # get some one lora model
    def _maybe_get_lora(self, request) -> Optional[LoRARequest]:
        if request.model == self.served_model:
            return
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError("The model `{request.model}` does not exist.")
