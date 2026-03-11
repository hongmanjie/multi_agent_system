# ref: modelscope/utils/import_utils.py
import importlib
import logging
import os
from importlib import import_module
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any

from xt_maas.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


class LazyImportModule(ModuleType):
    def __init__(self,
                 name,
                 module_file,
                 import_structure,
                 module_spec=None,
                 extra_objects=None,
                 try_to_pre_import=False,
                 extra_import_func=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure
        self._extra_import_func = extra_import_func
        if try_to_pre_import:
            self._try_to_import()

    def _try_to_import(self):
        for sub_module in self._class_to_module.keys():
            try:
                getattr(self, sub_module)
            except Exception as e:
                logger.warning(
                    f'pre load module {sub_module} error, please check {e}')

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        elif self._extra_import_func is not None:
            value = self._extra_import_func(name)
            if value is None:
                raise AttributeError(
                    f'module {self.__name__} has no attribute {name}')
        else:
            raise AttributeError(
                f'module {self.__name__} has no attribute {name}')

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            # module_name_full = self.__name__ + '.' + module_name
            return importlib.import_module('.' + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f'Failed to import {self.__name__}.{module_name} because of the following error '
                f'(look up to see its traceback):\n{e}') from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__,
                                self._import_structure)
