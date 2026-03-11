from typing import TYPE_CHECKING

from xt_maas.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .qwen2_5_vl import Qwen2_5VL_Model
else:
    _import_structure = {
        'qwen2_5_vl': ['Qwen2_5VL_Model']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
