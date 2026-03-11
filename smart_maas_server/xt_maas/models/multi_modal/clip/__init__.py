from typing import TYPE_CHECKING

from xt_maas.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .dfn5b_clip import DFN5B_CLIP
else:
    _import_structure = {
        'dfn5b_clip': ['DFN5B_CLIP']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
