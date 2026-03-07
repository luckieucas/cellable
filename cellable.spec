# -*- mode: python -*-
# vim: ft=python

import os
import os.path as osp
import sys

import osam._models.yoloworld.clip


sys.setrecursionlimit(5000)  # required on Windows

MODEL_BUNDLE = os.environ.get("CELLABLE_MODEL_BUNDLE", "efficientsam_accuracy").lower()
EXCLUDE_CELLPOSE = os.environ.get("CELLABLE_EXCLUDE_CELLPOSE", "1") == "1"
USE_STRIP = os.environ.get("CELLABLE_STRIP", "0") == "1"

# full: bundle every ONNX model in labelme/models (offline, largest size)
# balanced: bundle EfficientSAM + SAM-B
# lite: bundle EfficientSAM speed + accuracy
# efficientsam_accuracy: bundle only EfficientSAM (accuracy) vits encoder/decoder
_MODEL_BUNDLES = {
    "balanced": [
        "efficient_sam_vitt_encoder.onnx",
        "efficient_sam_vitt_decoder.onnx",
        "efficient_sam_vits_encoder.onnx",
        "efficient_sam_vits_decoder.onnx",
        "sam_vit_b_encoder.onnx",
        "sam_vit_b_decoder.onnx",
    ],
    "lite": [
        "efficient_sam_vitt_encoder.onnx",
        "efficient_sam_vitt_decoder.onnx",
        "efficient_sam_vits_encoder.onnx",
        "efficient_sam_vits_decoder.onnx",
    ],
    "efficientsam_accuracy": [
        "efficient_sam_vits_encoder.onnx",
        "efficient_sam_vits_decoder.onnx",
    ],
    "esam_accuracy": [
        "efficient_sam_vits_encoder.onnx",
        "efficient_sam_vits_decoder.onnx",
    ],
}

model_datas = []
if MODEL_BUNDLE == "full":
    model_datas.append(("labelme/models/*.onnx", "labelme/models"))
else:
    for filename in _MODEL_BUNDLES.get(MODEL_BUNDLE, _MODEL_BUNDLES["efficientsam_accuracy"]):
        path = osp.join("labelme", "models", filename)
        if osp.exists(path):
            model_datas.append((path, "labelme/models"))
        else:
            print(f"[cellable.spec] model missing, skipped: {path}")

hiddenimports = [
    "osam._models.yoloworld.clip",
    "em_util",
    "gdown",
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
]
excludes = []
if EXCLUDE_CELLPOSE:
    excludes.extend(
        [
            "cellpose",
            "torch",
            "torchvision",
            "torchaudio",
            "numba",
            "llvmlite",
        ]
    )
else:
    # CellPose is imported dynamically in code; include explicitly for full builds.
    hiddenimports.extend(["cellpose", "cellpose.models", "cellpose.io"])


a = Analysis(
    ['labelme/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons/*', 'labelme/icons'),
        ('labelme/translate/*.qm', 'translate'),
        (
            osp.join(
                osp.dirname(osam._models.yoloworld.clip.__file__),
                "bpe_simple_vocab_16e6.txt.gz",
            ),
            'osam/_models/yoloworld/clip',
        ),
    ] + model_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    name='cellable',
    debug=False,
    strip=USE_STRIP,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon='labelme/icons/icon.icns',
    exclude_binaries=True,
    argv_emulation=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=USE_STRIP,
    upx=False,
    name='cellable',
)
app = BUNDLE(
    coll,
    name='Cellable.app',
    icon='labelme/icons/icon.icns',
    bundle_identifier='org.cellable.cellable',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleDisplayName': 'Cellable',
        'CFBundleName': 'Cellable',
    },
)
