import os
import sys
import json
from zhilight import CPMBee, LLaMA

def load_model(cls = CPMBee, model_path = "/mnt/models", memory = 60 << 30):
    with open(f"{model_path}/config.json") as f:
        model_config = json.load(f)
    model_config["new_vocab"] = True
    model_config["scale_weights"] = True
    model_config["weight_transposed"] = False
    model_pt = True
    model_file = f"{model_path}/model.pt"
    if os.path.isfile(f"{model_path}/model.ckpt"):
        model_pt = False
        model_config["scale_weights"] = False
        model_config["weight_transposed"] = True
        model_file = f"{model_path}/model.ckpt"

    sys.stderr.write("loading model.\n")
    model = cls(
        model_path = model_file,
        vocab_path = f"{model_path}/vocabs.txt",
        device_id = -1,
        memory_limit = memory,
        model_config = model_config,
        load_model = not model_pt,
    )

    if model_pt:
        model.load_model_pt(model_file)
    sys.stderr.write("model loaded.\n")

    return model

def main():
    datas = [
        {
            "input": "北京是中国的首都。",
            "prompt": "中译英",
            "<ans>": "Beijing is the capital of China.",
        },
        "李彦宏是百度公司创始人",
        "如何评价知乎公司？请写200字评价",
    ]
    #model = load_model()
    model = load_model(LLaMA)
    #res = model.inference(datas[1], fetch_tokens_level = 1, prepend_input = False)
    #res =model.random_search(datas[1], fetch_tokens_level = 1, prepend_input = True)
    res = model.inference(datas[1:3], fetch_tokens_level = 1, prepend_input = True, return_new = True)
    print(json.dumps(res, ensure_ascii = False))

if __name__ == "__main__":
    main()