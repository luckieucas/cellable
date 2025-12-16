# -*- mode: python -*-
# vim: ft=python

import os.path as osp
import sys

import osam._models.yoloworld.clip

sys.setrecursionlimit(5000)  # required on Windows

a = Analysis(
    ['labelme/__main__.py'],
    pathex=[],  # 如果 em_util 已安装，可以留空
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
        # 添加 em_util 相关导入
        'em_util',
        'em_util.seg',
        'em_util.seg.iou',
        'em_util.seg.basic',
        'em_util.io',
        'em_util.io.io',
        'em_util.io.arr',
        'em_util.io.box',
        'em_util.io.image',
        'em_util.io.set',
        'em_util.io.skel',
        'em_util.io.tile',
        'em_util.eval',
        'em_util.eval.seg',
        'em_util.ng',
        'em_util.ng.ng_dataset',
        'em_util.ng.ng_layer',
        'em_util.vast',
        'em_util.vast.io',
        'em_util.vast.meta',
        'em_util.video',
        'em_util.video.Basic_Shot_Detection',
        'em_util.video.html',
        'em_util.video.html.html_base',
        'em_util.video.html.save_data',
        'em_util.video.html.shot_detection',
        'em_util.cluster',
        'em_util.cluster.slurm',
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
