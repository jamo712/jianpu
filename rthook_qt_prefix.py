# -*- coding: utf-8 -*-
"""在 PyInstaller 自带钩子之前写入 qt.conf。

Qt5 把 qt.conf 按 Latin-1 保存。安装目录若是中文，绝对路径会变成问号，
谱面浏览器找不到资源，主程序会退出。
这里只写相对程序目录的英文路径，中文目录名由 Qt 用 Unicode 拼接。
拼接前先把当前目录设回程序所在目录，避免系统 ANSI 接口把中文目录写坏。
"""
import ctypes
import os
import sys

if getattr(sys, "frozen", False) and sys.platform == "win32":
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    ctypes.windll.kernel32.SetCurrentDirectoryW(exe_dir)

if getattr(sys, "frozen", False):
    from _pyi_rth_utils import qt as qt_rth_utils

    qt_rth_utils.create_embedded_qt_conf("PyQt5", "_internal/PyQt5/Qt5")
