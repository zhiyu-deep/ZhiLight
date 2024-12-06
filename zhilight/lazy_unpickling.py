import io
import os
import zipfile
import torch
import pickle
from torch.serialization import (
    _maybe_decode_ascii,
    StorageType,
)


class LazyUnpickleStorage:
    def __init__(self, file_path):
        self.zip_file = torch._C.PyTorchFileReader(open(file_path, "rb"))
        self.py_zip_file = zipfile.ZipFile(file_path, "r")
        data_pkl_files = [
            fn for fn in self.py_zip_file.namelist() if fn.endswith("data.pkl")
        ]
        if len(data_pkl_files) != 1:
            raise ValueError("not a pt file!")
        self.py_zip_basedir = os.path.dirname(data_pkl_files[0])
        self.placeholders, self.metadata = self._load(self.zip_file)

    def __getitem__(self, name):
        placeholder = self.placeholders[name]
        key = str(placeholder.flatten()[0].item())
        dtype, nbytes, _ = self.metadata[key]
        data = torch.empty_like(placeholder, dtype=dtype)
        data.set_(self.load_tensor_storage(nbytes, key, dtype))
        return data.view(placeholder.size()).contiguous()

    def keys(self):
        return self.placeholders.keys()

    @staticmethod
    def _load(zip_file, pickle_file="data.pkl", **pickle_load_args):
        loaded_storages = {}
        loaded_index = {}

        def load_index(dtype, numel, key, location):
            # store the key to returned tensor. each tensor only takes 8 bytes.
            loaded_storages[key] = torch.tensor([int(key)]).storage()
            loaded_index[key] = (dtype, numel, location)

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            assert (
                typename == "storage"
            ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            storage_type, key, location, numel = data
            if storage_type is torch.UntypedStorage:
                dtype = torch.uint8
            else:
                dtype = storage_type.dtype

            if key not in loaded_storages:
                nbytes = numel * torch._utils._element_size(dtype)
                load_index(dtype, nbytes, key, _maybe_decode_ascii(location))
                # load_tensor(dtype, 0, key, _maybe_decode_ascii(location))

            return loaded_storages[key]

        load_module_mapping: Dict[str, str] = {
            # See https://github.com/pytorch/pytorch/pull/51633
            "torch.tensor": "torch._tensor"
        }

        # Need to subclass Unpickler instead of directly monkey-patching the find_class method
        # because it's marked readonly in pickle.
        # The type: ignore is because mypy can't statically determine the type of this class.
        class UnpicklerWrapper(pickle.Unpickler):  # type: ignore[name-defined]
            # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
            # Lets us override the imports that pickle uses when unpickling an object.
            # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
            def find_class(self, mod_name, name):
                if type(name) is str and "Storage" in name:
                    try:
                        return StorageType(name)
                    except KeyError:
                        pass
                mod_name = load_module_mapping.get(mod_name, mod_name)
                return super().find_class(mod_name, name)

        # Load the data (which may in turn use `persistent_load` to load tensors)
        data_file = io.BytesIO(zip_file.get_record(pickle_file))

        unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()

        torch._utils._validate_loaded_sparse_tensors()

        return result, loaded_index

    def load_tensor_storage(self, nbytes, key, dtype):
        name = f"{self.py_zip_basedir}/data/{key}"
        storage = (
            torch.frombuffer(self.py_zip_file.open(name).read(), dtype=dtype)
            .storage()
            .untyped()
        )
        return storage
