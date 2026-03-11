from typing import TYPE_CHECKING

from xt_maas.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .qwen2_5 import Qwen2_5_TransformerModel, Qwen2_5_VllmModel
    from .qwen3 import Qwen3_TransformerModel
else:
    _import_structure = {
        'qwen2_5': ['Qwen2_5_TransformerModel', 'Qwen2_5_VllmModel'],
        'qwen3': ['Qwen3_TransformerModel']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
