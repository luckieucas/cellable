# -*- mode: python -*-
# vim: ft=python

import os.path as osp
import sys

import osam._models.yoloworld.clip


sys.setrecursionlimit(5000)  # required on Windows


a = Analysis(
    ['labelme/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons/*', 'labelme/icons'),
        ('labelme/translate/*.qm', 'translate'),
        # Bundle AI models for offline use (downloaded via download_models.py)
        ('labelme/models/*.onnx', 'labelme/models'),
        (
            osp.join(
                osp.dirname(osam._models.yoloworld.clip.__file__),
                "bpe_simple_vocab_16e6.txt.gz",
            ),
            'osam/_models/yoloworld/clip',
        ),
    ],
    hiddenimports=[
        'osam._models.yoloworld.clip',
        'em_util',
        'gdown',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    name='cellable',
    debug=False,
    strip=False,
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
    strip=False,
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
