"""Sequence and its related classes."""
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs."""
    logprob: float
    decoded_token: Optional[str] = None


PromptLogprobs = List[Optional[Dict[int, Logprob]]]
SampleLogprobs = List[Dict[int, Logprob]]


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Args:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    time_in_queue: Optional[float] = None
    finished_time: Optional[float] = None
    input_tokens_num: Optional[int] = None
    output_tokens_num: Optional[int] = None