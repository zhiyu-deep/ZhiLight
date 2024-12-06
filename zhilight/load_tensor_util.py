# coding=utf-8
# mainly for debug to fast reload model

import gc
import glob
import numpy as np
import os
import pathlib
import pickle
import re
import time
import torch
# need py3.8
from multiprocessing.shared_memory import SharedMemory
from zhilight.loader import LLaMALoader

try:
    from multiprocessing.resource_tracker import unregister
except ModuleNotFoundError:
    def unregister(name, t):
        pass


def load_pt_to_dict(model_dir):
    state_dict = {}
    if os.path.isfile(f'{model_dir}'):
        state_dict = torch.load(f'{model_dir}', map_location='cpu')
    elif os.path.isfile(f'{model_dir}/pytorch_model.pt'):
        state_dict = torch.load(f'{model_dir}/pytorch_model.pt', map_location='cpu')
    else:
        pt_files = sorted(glob.glob(f'{model_dir}/pytorch_model*.bin'))
        if not pt_files:
            pt_files = sorted(glob.glob(f"{model_dir}/*.pt"))
        if not pt_files:
            raise ValueError(f'No checkpoint found in: {model_dir}')
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu"))
    return state_dict


def get_dir_or_file_size(dir):
    if os.path.isfile(dir):
        return os.stat(dir).st_size
    return sum(file.stat().st_size for file in pathlib.Path(dir).rglob('*'))


np_dict = {}  # global to delay gc (speed up load a few seconds)
shm_mems = []
def load_pt_with_cache(model_path: str, load_fn=load_pt_to_dict):
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
        mem_limit = int(f.read())
    shm_size = get_dir_or_file_size('/dev/shm')
    model_size = get_dir_or_file_size(model_path)

    model_path_e = 'NPC_' + model_path.replace("/", "_").replace(".", "_")
    d = f"/dev/shm/{model_path_e}"
    shape_filename = f"{d}_all_shape"
    if not os.path.exists(shape_filename):
        if model_size + shm_size > mem_limit:
            print('Clean cache')
            os.system(f'rm -rf /dev/shm/NPC_*')
        print(f'Building shared_memory cache for "{model_path}" ...\n')

        state_dict: dict = load_fn(model_path)
        state_dict = LLaMALoader.convert_quant_dict(state_dict)
        state_list = [(name, param) for name, param in state_dict.items()]
        state_dict.clear()

        shape_dict = {}
        while state_list:
            name, param = state_list.pop(0)
            print('\rWrite', f"/dev/shm/..._{name}                      ", end="")
            if param.dtype == torch.bfloat16:
                param = param.view(torch.int16)
            a = param.numpy()
            with open(f"/dev/shm/{model_path_e}_{name}", 'wb') as f:
                a.tofile(f)
            shape_dict[name] = a.shape
            del a
            # print(gc.get_referrers(param))
            # print(sys.getrefcount(param))
            del param
            gc.collect()
        with open(shape_filename, "wb") as f:
            f.write(pickle.dumps(shape_dict))
        print("\nDone cache to shared_memory.")

    with open(shape_filename, "rb") as f:
        shape_dict = pickle.load(f)  # type: dict[str, tuple]

    status = 0
    # for name, shape in shape_dict.items():
    #     import subprocess
    #     status = subprocess.Popen(["lsof", f"/dev/shm/{model_path_e}_{name}"], stdout=subprocess.DEVNULL).wait()
    #     break
    print("Load state_dict from shared memory.")
    global np_dict
    for name, shape in shape_dict.items():
        if 'inv_freq' in name:
            continue
        shm = SharedMemory(f"{model_path_e}_{name}")
        shm_mems.append(shm)
        unregister(shm._name, "shared_memory")
        b = np.frombuffer(shm.buf, dtype="float16" if 'quant_weight' not in name else "int8")
        # print(name)
        # print(f"std: {np.std(b)}, max: {np.max(b)}, min: {np.min(b)}")
        b1 = b.reshape(shape)
        np_dict[name] = b1
        shm._mmap = None
    return np_dict


def load_llama_cache(model, model_path):
    state_dict = load_pt_with_cache(model_path)
    state_dict = {LLaMALoader._replace_name(name): p for name, p in state_dict.items()}
    return model._base._model.load_state_dict(state_dict)


def load_cpmbee_cache(model, model_path):
    from zhilight.convert import chname
    state_dict = load_pt_with_cache(model_path)
    state_dict = {chname("cpm_bee", name): p for name, p in state_dict.items()}
    return model._base._model.load_state_dict(state_dict)
