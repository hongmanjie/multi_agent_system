from typing import TYPE_CHECKING

from xt_maas.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .yolo_person import YOLOv12_ONNX_PERSON
    from .yolo_helmet import YOLOv12_ONNX_HELMET
    from .yolo_pose import YOLOv12_ONNX_POSE
    from .grounding_dino import GroundingDINO
    from .damo_yolo import DAMOYOLO_ONNX_HELMET
    from .sam3 import SAM3Model

else:
    _import_structure = {'yolo_person': ['YOLOv12_ONNX_PERSON'],
                         'yolo_helmet': ['YOLOv12_ONNX_HELMET'],
                         'yolo_pose': ['YOLOv12_ONNX_POSE'],
                         'grounding_dino': ['GroundingDINO'],
                         'damo_yolo': ['DAMOYOLO_ONNX_HELMET'],
                         'sam3': ['SAM3Model']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
