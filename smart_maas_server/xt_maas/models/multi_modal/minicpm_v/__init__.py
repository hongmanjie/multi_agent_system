from typing import TYPE_CHECKING

from xt_maas.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .minicpmv_4_5 import MiniCPM_V_Model
else:
    _import_structure = {
        'minicpmv_4_5': ['MiniCPM_V_Model']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
