# download_models.py
import os
import os.path as osp
import gdown
import shutil

# 模型下载配置
MODEL_URLS = {
    # SegmentAnything models
    'sam_vit_b_encoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx',
        'md5': '80fd8d0ab6c6ae8cb7b3bd5f368a752c',
    },
    'sam_vit_b_decoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx',
        'md5': '4253558be238c15fc265a7a876aaec82',
    },
    'sam_vit_l_encoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx',
        'md5': '080004dc9992724d360a49399d1ee24b',
    },
    'sam_vit_l_decoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx',
        'md5': '851b7faac91e8e23940ee1294231d5c7',
    },
    'sam_vit_h_encoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx',
        'md5': '958b5710d25b198d765fb6b94798f49e',
    },
    'sam_vit_h_decoder.onnx': {
        'url': 'https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.decoder.onnx',
        'md5': 'a997a408347aa081b17a3ffff9f42a80',
    },
    # EfficientSAM models
    'efficient_sam_vitt_encoder.onnx': {
        'url': 'https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx',
        'md5': '2d4a1303ff0e19fe4a8b8ede69c2f5c7',
    },
    'efficient_sam_vitt_decoder.onnx': {
        'url': 'https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx',
        'md5': 'be3575ca4ed9b35821ac30991ab01843',
    },
    'efficient_sam_vits_encoder.onnx': {
        'url': 'https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx',
        'md5': '7d97d23e8e0847d4475ca7c9f80da96d',
    },
    'efficient_sam_vits_decoder.onnx': {
        'url': 'https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx',
        'md5': 'd9372f4a7bbb1a01d236b0508300b994',
    },
}

# 创建模型目录
models_dir = 'labelme/models'
os.makedirs(models_dir, exist_ok=True)

print(f"Downloading models to {models_dir}...")

# 下载所有模型文件
for filename, config in MODEL_URLS.items():
    filepath = osp.join(models_dir, filename)
    
    # 如果文件已存在，跳过
    if osp.exists(filepath):
        print(f"✓ {filename} already exists, skipping...")
        continue
    
    try:
        print(f"Downloading {filename}...")
        # 先下载到临时文件
        temp_path = gdown.cached_download(
            url=config['url'],
            md5=config['md5'],
        )
        
        # 复制到目标目录
        shutil.copy2(temp_path, filepath)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print(f"\nAll models downloaded to {models_dir}")