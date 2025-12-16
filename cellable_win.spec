# -*- mode: python -*-
# vim: ft=python

import os.path as osp
import sys

import osam._models.yoloworld.clip

sys.setrecursionlimit(5000)  # required on Windows

a = Analysis(
    ['labelme/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('labelme/config/default_config.yaml', 'labelme/config'),
        ('labelme/icons', 'labelme/icons'),
        ('labelme/translate/*.qm', 'translate'),
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
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cellable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 设置为 True 可以看到控制台输出（调试用）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='labelme/icons/icon.ico',
)
