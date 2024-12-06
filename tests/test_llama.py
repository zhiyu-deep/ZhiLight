import os

os.environ[
    "LD_LIBRARY_PATH"
] = "/home/jeeves/.local/lib/python3.10/site-packages/nvidia/cublas/lib/"

from zhilight import LLaMA, QuantConfig, QuantType
from zhilight.llama import LLaMAModelConfig

try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
except:
    LlamaTokenizer = None
    LlamaForCausalLM = None
import numpy as np
import time
import torch


def main(model_path="llama-13b-hf"):
    assert LlamaTokenizer, "llama tokenizer load failed, pip install transformer"
    print(f"Test {model_path}")
    quant_config = QuantConfig(type=QuantType.AutoInt8)
    quant_config = None
    load_model_pt = True
    origin_model_dir = f"/local/llama/{model_path}"
    model_config_13b: LLaMAModelConfig = {
        "num_layers": 40,
        "dim_model": 5120,
        "num_heads": 40,
        "dim_head": 128,
        "dim_ff": 13824,
        "vocab_size": 32000,
    }

    model_config_65b: LLaMAModelConfig = {
        "num_layers": 80,
        "dim_model": 8192,
        "num_heads": 64,
        "dim_head": 128,
        "dim_ff": 22016,
        "vocab_size": 32000,
    }

    if model_path == "llama-65b-hf":
        model = LLaMA(
            f"{origin_model_dir}.ckpt",
            origin_model_dir,
            -1,
            memory_limit=70 << 30,
            model_config=model_config_65b,
            quant_config=quant_config,
            load_model=not load_model_pt,
            weight_transposed=not load_model_pt,
        )
    else:
        model = LLaMA(
            f"{origin_model_dir}.ckpt",
            origin_model_dir,
            -1,
            memory_limit=30 << 30,
            model_config=model_config_13b,
            quant_config=quant_config,
            load_model=not load_model_pt,
            weight_transposed=not load_model_pt,
        )

    if load_model_pt:
        model.load_model_pt(origin_model_dir)

    datas = ["Happy", "北京是", "夏天", "好", "May", "Cook", "Ten", "一个", "热天", "优秀"]

    print(model.inference(datas, max_length=30))  # inference batch

    for data in datas:
        print(model.inference(data, max_length=10))
        print(model.random_search(data))

    while True:
        s = input("输入：")
        if s:
            s = s.replace("\\n", "\n")
            print(model.inference(s, max_length=100))

    print(f"Done inference from zhilight")

    # logits, hiddens = model.get_logits(datas, return_hidden_states=True)
    # logits, hiddens = torch.tensor(logits), torch.tensor(hiddens)

    # hug
    print("Loading LlamaForCausalLM...")
    tokenizer = LlamaTokenizer.from_pretrained(origin_model_dir)
    hug_model = LlamaForCausalLM.from_pretrained(origin_model_dir, device_map="auto")
    hug_model.eval()
    print("Done loading LlamaForCausalLM")
    inputs = tokenizer(datas[0], return_tensors="pt")
    while True:
        s = input("输入：")
        if not s:
            continue
        inputs = tokenizer(s, return_tensors="pt")

        t1 = time.time()
        generate_ids = hug_model.generate(
            inputs.input_ids, max_length=100
        )  # , num_beams=3, do_sample=True)
        hug_ans = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(
            f"{hug_ans}  tokens={len(generate_ids)}, avg={(time.time() - t1) / len(generate_ids)}"
        )

    hug = hug_model(inputs.input_ids, return_dict=True, output_hidden_states=True)
    hug_logits = hug.logits
    hug_hiddens = hug.hidden_states
    for i in range(33):
        print(i, (hiddens[0][i] - hug_hiddens[i][0]).abs().max())
    print("logits", (logits - hug_logits).abs().max())


if __name__ == "__main__":
    # main()
    main("llama-13b-hf")
    # main('llama-65b-hf')
