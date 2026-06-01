# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


repo_root = Path(SPECPATH).parent
app_icon_path = repo_root / "packaging" / ("app_icon.icns" if sys.platform == "darwin" else "app_icon.ico")

datas = [
    (str(repo_root / "webapp" / "presentation" / "templates"), "webapp/presentation/templates"),
    (str(repo_root / "webapp" / "presentation" / "static"), "webapp/presentation/static"),
    (str(repo_root / "pipelines" / "OpenSim_Setup"), "pipelines/OpenSim_Setup"),
    (str(repo_root / "pipelines" / "MarkerAugmenter"), "pipelines/MarkerAugmenter"),
    (str(repo_root / "pipelines" / "models"), "pipelines/models"),
    (str(repo_root / "images"), "images"),
]
binaries = collect_dynamic_libs("opensim") + collect_dynamic_libs("onnxruntime")
hiddenimports = (
    collect_submodules("flask_socketio")
    + collect_submodules("engineio")
    + collect_submodules("socketio")
    + collect_submodules("webapp")
    + collect_submodules("pipelines")
)

a = Analysis(
    [str(repo_root / "webapp" / "main.py")],
    pathex=[str(repo_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Human3DMotion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(app_icon_path),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Human3DMotion",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Human3DMotion.app",
        icon=str(app_icon_path),
        bundle_identifier="com.Human3DMotion.app",
    )
