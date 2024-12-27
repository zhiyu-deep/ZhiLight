# coding=utf-8
# Author: spetrel@gmail.com

from .dev_config import *
from .qwen2_adapter import Qwen2Adapter
from .cohere_adapter import CohereAdapter


def _get_model_type(config: dict):
    model_type = config.get("model_type", "")
    if model_type:
        return model_type
    architectures = config.get("architectures", [""])
    if "minicpm" in architectures[0].lower():
        config["model_type"] = "cpm_dragonfly"
        return "cpm_dragonfly"
    return ""


class ModelAdapter:
    @staticmethod
    def adapt(config: dict):
        model_type = config.get("model_type", "")
        if model_type == "qwen2":
            Qwen2Adapter.adapt(config)
        elif model_type == "cohere":
            CohereAdapter.adapt(config)

        if get_int_env(CHUNKED_PREFILL) == 1:
            set_env("DUAL_STREAM_THRESHOLD", 100)

        ModelAdapter.adapt_gptq(config)

        return config

    @staticmethod
    def adapt_gptq(config: dict):
        quant_config = config.get("quantization_config", {})
        if (
                quant_config
                and quant_config.get("desc_act", False)
                and config.get("bfloat16", False)
                and not config.get("force_half", False)
        ):
            print("WARNING: force convert to half dtype for using GPTQ kernel")
            config["bfloat16"] = False
            config["force_half"] = True
