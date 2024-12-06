import time
import torch

from zhilight.loader import LLaMALoader
from zhilight.lazy_unpickling import LazyUnpickleStorage


def main():
    rtol, atol = (1e-5, 3e-5)
    state_dict = torch.load("/home/gnap/Downloads/caterpillar-7b/pytorch_model.bin")
    lazy_state_dict = LazyUnpickleStorage(
        "/home/gnap/Downloads/caterpillar-7b/pytorch_model.bin"
    )
    # lazy_state_dict = LLaMALoader(
    #    "/home/gnap/Downloads/caterpillar-7b", True
    # )._state_dict
    for name in lazy_state_dict.keys():
        x = lazy_state_dict[name]
        y = state_dict[name]
        print(x)
        # del x
        print(y)
        assert torch.allclose(x, y, rtol=rtol, atol=atol)
    time.sleep(300)


if __name__ == "__main__":
    main()
