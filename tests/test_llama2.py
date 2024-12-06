import os
import time

from zhilight import LLaMA
from zhilight.llama import LLaMAModelConfig

try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
except:
    LlamaTokenizer = None
    LlamaForCausalLM = None
import numpy as np
import time
import torch

def load_model(ver="7b", load_model_pt=False):
    t0 = time.time()
    assert LlamaTokenizer, "llama tokenizer load failed, pip install transformer"
    model_dir = f"/mnt/data/user/tc_agi/user/zhaoweilin/zhilight-llama-2-{ver}"

    model_config_7b: LLaMAModelConfig = {
        "num_layers": 32,
        "dim_model": 4096,
        "num_heads": 32,
        "dim_head": 128,
        "dim_ff": 11008,
        "vocab_size": 32000,
        "eps": 1e-5,
    }

    model_config_13b: LLaMAModelConfig = {
        "num_layers": 40,
        "dim_model": 5120,
        "num_heads": 40,
        "dim_head": 128,
        "dim_ff": 13824,
        "vocab_size": 32000,
        "eps": 1e-5,
    }

    model_config_70b: LLaMAModelConfig = {
        "num_layers": 80,
        "dim_model": 8192,
        "num_heads": 64,
        "dim_head": 128,
        "dim_ff": 28672,
        "vocab_size": 32000,
        "eps": 1e-5,
        "num_kv_heads": 8,
    }

    if ver.startswith("7b"):
        model = LLaMA(
            f"{model_dir}/model.ckpt",
            model_dir,
            -1,
            memory_limit = 30 << 30,
            model_config=model_config_7b,
        )
    elif ver.startswith("13b"):
        model = LLaMA(
            f"{model_dir}/model.ckpt",
            model_dir,
            0,
            memory_limit = 70 << 30,
            model_config=model_config_13b,
        )
    elif ver.startswith("70b"):
        model = LLaMA(
            f"{model_dir}/model.ckpt",
            model_dir,
            -1,
            memory_limit = 76 << 30,
            model_config=model_config_70b,
            load_model=not load_model_pt,
            weight_transposed=not load_model_pt,
        )
    else:
        raise ValueError(f"Unknown version {ver}")

    if load_model_pt:
        model.load_model_pt(f"{model_dir}")
    print(f"model load finished in {time.time() - t0} seconds")
    return model

def compare_batch_diff(model, test_num=15, single_base=True, bench_batch=True, bench_dyn_batch=True):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/sentence.txt') as f:
        sentences = f.readlines()
    datas = ["Translate to english, no explanation.\n" + s.split(".", maxsplit=1)[-1].strip() for s in sentences]
    datas = datas[0:test_num]
    max_length = 100

    single_scores = [0.] * len(datas)
    single_answers = []
    if single_base:
        print("\n\n############### single beam search ###############")
        single_scores = []
        for d in datas:
            res = model.inference(d, max_length=max_length, return_new=True)
            single_scores.append(res['score'])
            single_answers.append(res['result'])
            print(res)

    batch_size = 10
    batch_scores = []
    batch_answers = []
    batch_time = 1000
    if bench_batch:
        print("\n\n############### batch beam search ###############")
        st = time.time()
        p_list = [datas[i:i + batch_size] for i in range(0, len(datas), batch_size)]
        for p in p_list:
            b_out = model.inference(p, max_length=max_length, return_new=True)
            for res in b_out:
                batch_scores.append(res['score'])
                batch_answers.append(res['result'])
                print(res)
        batch_time = int(1000 * (time.time() - st))
        max_score_diff = np.abs(np.array(batch_scores) - np.array(single_scores)).max()
        num_answer_diff = sum([a != b for a, b in zip(single_answers, batch_answers)])
        print(f"batch_scores: {batch_scores}")
        print(f"### batch take {batch_time}ms. max_score_diff={max_score_diff}, num_answer_diff={num_answer_diff}")

    if bench_dyn_batch:
        print("\n\n############### dynamic batch beam search ###############")
        st = time.time()
        b_out = model.dynamic_batch_inference(datas, max_batch=batch_size, max_length=max_length)
        dynamic_time = int(1000 * (time.time() - st))
        for res in b_out:
            print(res)
        dynamic_scores = [res['score'] for res in b_out]
        dynamic_answers = [res['result'] for res in b_out]
        max_score_diff = np.abs(np.array(dynamic_scores) - np.array(single_scores)).max()
        num_answer_diff = sum([a != b for a, b in zip(single_answers, dynamic_answers)])
        for a, b in zip(single_answers, dynamic_answers):
            if a != b:
                print("Exp: ", a)
                print("Got: ", b)
        print(f"dynamic_scores: {dynamic_scores}")
        print(f"### dyn batch take {dynamic_time}ms; SpeedUp={batch_time / dynamic_time}")
        print(f"max_score_diff={max_score_diff}, num_answer_diff={num_answer_diff}")

def main(ver="7b"):
    model = load_model(ver, False)
    # compare_batch_diff(model)

    for data in datas:
        print(model.inference(data, max_length=10))
        print(model.random_search(data))

if __name__ == "__main__":
    # main("7b")
    # main("13b")
    # main("70b")
    main("7b-chat")
    # main("13b-chat")
    # main("70b-chat")
