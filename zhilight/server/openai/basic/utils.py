import os
import socket
import uuid
import subprocess
import argparse
from fastapi import Request
from typing import List, Union

import psutil
import torch
import asyncio
from functools import partial
from typing import (
    Awaitable,
    Callable,
    TypeVar,
)
from typing import Optional

from zhilight.server.openai.basic.logger import init_logger
import warnings

T = TypeVar("T")
logger = init_logger(__name__)
QUOTATION_MARKS="“”‘’'\""

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8_e5m2": torch.uint8,
}

def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """
    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper


def get_ip() -> str:
    host_ip = os.environ.get("HOST_IP")
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"


def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

def pad_to_max_length(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))

def make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = [pad_to_max_length(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)


def maybe_expand_dim(tensor: torch.Tensor,
                     target_dims: int,
                     size: int = 1) -> torch.Tensor:
    """Expand the tensor to the target_dims."""
    if tensor.ndim < target_dims:
        tensor = tensor.view(-1, *([size] * (target_dims - tensor.ndim)))
    return tensor

def _strip_quotations(text: str):
    if isinstance(text, str):
        return text.strip(QUOTATION_MARKS)
    return text

def parse_zhilight_version(version):
    version = _strip_quotations(version)
    v = _get_zhilight_version()
    if version is None or v == version:
        return v
    logger.info(f"current zhilight={v}, changed to zhilight={version}.")
    subprocess.run([
        'pip', 
        'install', 
        f"zhilight=={version}", 
        '--force-reinstall', 
        '--timeout=1000',
        '--index=https://pypi.tuna.tsinghua.edu.cn/simple'])
    v2 = _get_zhilight_version()
    assert v2==version, "force reinstall zhilight={version} failed"
    return version

def _get_zhilight_version():
    result = subprocess.run(['pip', 'show', 'zhilight'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    version = None
    for line in lines:
        tokens = line.split(':')
        if len(tokens) == 2:
            if 'version' == tokens[0].strip().lower():
                version = tokens[1].strip()
                break
    assert version is not None, "must install zhilight."
    return version

def register_environs(envs: List[str]):
    for env in envs:
        env = _strip_quotations(env)
        items = env.split(';')
        for item in items:
            tokens = item.split('=')
            if len(tokens) == 2:
                key = tokens[0].strip()
                val = tokens[1].strip()
                if len(key) > 0:
                    os.environ[key] = val
                    logger.info(f"register env {key}={val}")

def get_options_info(parser):
    """从argparse.ArgumentParser对象中获取所有actions的描述。"""
    infos = {
        "description": "CPM-Server OpenAI Compatible Interface Options",
        "options": {},
    }
    for action in parser._actions:
        # 忽略帮助和版本等内置actions
        if action.dest is not argparse.SUPPRESS:
            infos["options"][f"--{action.dest.replace('_', '-')}"] = action.help
    return infos

def force_install_packages(packages: List[str]):
    for p in packages:
        p = _strip_quotations(p)
        subprocess.run([
            'pip',
            'install',
            p.strip(),
            '--force-reinstall',
            '--timeout=1000',
            '--index=https://pypi.tuna.tsinghua.edu.cn/simple'
        ])

def get_traceid(request: Request):
    traceid = request.headers.get('x-b3-traceid')
    if traceid is None:
        traceid = random_uuid()
    return traceid