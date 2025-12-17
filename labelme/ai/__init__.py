import os
import os.path as osp
import sys
import gdown

from .efficient_sam import EfficientSam
from .segment_anything_model import SegmentAnythingModel
from .segment_all import CellPose, nnUNet
from .text_to_annotation import get_rectangles_from_texts  # NOQA: F401
from .text_to_annotation import get_shapes_from_annotations  # NOQA: F401
from .text_to_annotation import non_maximum_suppression  # NOQA: F401


def get_model_path(filename, url, md5):
    """
    优先从 labelme/models 目录加载模型文件，如果不存在则使用 gdown 下载。
    
    Args:
        filename: 模型文件名（如 'efficient_sam_vitt_encoder.onnx'）
        url: 模型下载 URL
        md5: 模型文件的 MD5 校验值
    
    Returns:
        模型文件的完整路径
    """
    # 尝试从多个可能的位置查找模型文件
    possible_paths = [
        # 打包后的 exe 中，模型文件在 labelme/models 目录
        osp.join(osp.dirname(osp.dirname(__file__)), 'models', filename),
        # 开发环境中，模型文件在 labelme/models 目录
        osp.join(osp.dirname(__file__), '..', 'models', filename),
        # 绝对路径（如果设置了环境变量）
        osp.join(os.getcwd(), 'labelme', 'models', filename),
    ]
    
    # 检查 PyInstaller 打包后的资源路径
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时目录
        bundled_path = osp.join(sys._MEIPASS, 'labelme', 'models', filename)
        possible_paths.insert(0, bundled_path)
    
    # 按优先级查找文件
    for path in possible_paths:
        if osp.exists(path) and osp.isfile(path):
            return path
    
    # 如果都找不到，使用 gdown 下载到缓存目录
    return gdown.cached_download(url=url, md5=md5)

class SegmentAnythingModelVitB(SegmentAnythingModel):
    name = "SegmentAnything (speed)"

    def __init__(self):
        super().__init__(
            encoder_path=get_model_path(
                filename='sam_vit_b_encoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx",  # NOQA
                md5="80fd8d0ab6c6ae8cb7b3bd5f368a752c",
            ),
            decoder_path=get_model_path(
                filename='sam_vit_b_decoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx",  # NOQA
                md5="4253558be238c15fc265a7a876aaec82",
            ),
        )


class SegmentAnythingModelVitL(SegmentAnythingModel):
    name = "SegmentAnything (balanced)"

    def __init__(self):
        super().__init__(
            encoder_path=get_model_path(
                filename='sam_vit_l_encoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx",  # NOQA
                md5="080004dc9992724d360a49399d1ee24b",
            ),
            decoder_path=get_model_path(
                filename='sam_vit_l_decoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx",  # NOQA
                md5="851b7faac91e8e23940ee1294231d5c7",
            ),
        )


class SegmentAnythingModelVitH(SegmentAnythingModel):
    name = "SegmentAnything (accuracy)"

    def __init__(self):
        super().__init__(
            encoder_path=get_model_path(
                filename='sam_vit_h_encoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx",  # NOQA
                md5="958b5710d25b198d765fb6b94798f49e",
            ),
            decoder_path=get_model_path(
                filename='sam_vit_h_decoder.onnx',
                url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.decoder.onnx",  # NOQA
                md5="a997a408347aa081b17a3ffff9f42a80",
            )
        )


class EfficientSamVitT(EfficientSam):
    name = "EfficientSam (speed)"

    def __init__(self):
        super().__init__(
            encoder_path=get_model_path(
                filename='efficient_sam_vitt_encoder.onnx',
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx",  # NOQA
                md5="2d4a1303ff0e19fe4a8b8ede69c2f5c7",
            ),
            decoder_path=get_model_path(
                filename='efficient_sam_vitt_decoder.onnx',
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx",  # NOQA
                md5="be3575ca4ed9b35821ac30991ab01843",
            )
        )


class EfficientSamVitS(EfficientSam):
    name = "EfficientSam (accuracy)"

    def __init__(self):
        super().__init__(
            encoder_path=get_model_path(
                filename='efficient_sam_vits_encoder.onnx',
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",  # NOQA
                md5="7d97d23e8e0847d4475ca7c9f80da96d",
            ),
            decoder_path=get_model_path(
                filename='efficient_sam_vits_decoder.onnx',
                url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx",  # NOQA
                md5="d9372f4a7bbb1a01d236b0508300b994",
            )
        )


MODELS = [
    SegmentAnythingModelVitB,
    SegmentAnythingModelVitL,
    SegmentAnythingModelVitH,
    EfficientSamVitT,
    EfficientSamVitS,
    CellPose,
    nnUNet,
]
