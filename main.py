#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import ctypes
from ctypes import wintypes
import re
import json
import mimetypes
import socket
import subprocess
import shutil
import threading
import queue
from pathlib import Path
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

def _prepare_webengine():
    """打包后若沙箱或 GPU 进程异常，点击“打开谱面”会把主程序一起退出。"""
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    for flag in ("--disable-gpu", "--no-sandbox"):
        if flag not in flags.split():
            flags = (flags + " " + flag).strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
    if getattr(sys, "frozen", False):
        webengine = Path(sys._MEIPASS) / "PyQt5" / "Qt5" / "bin" / "QtWebEngineProcess.exe"
        if webengine.is_file():
            os.environ["QTWEBENGINEPROCESS_PATH"] = str(webengine)
        _stage_webengine_fallback()


def _stage_webengine_fallback():
    """把浏览器资源挂到程序旁边。

    安装目录是中文时，Qt 记录的资源路径会损坏。浏览器接着会到程序所在目录找这些文件。
    用硬链接过去，不额外占空间。
    """
    exe_dir = Path(sys.executable).resolve().parent
    root = Path(sys._MEIPASS) / "PyQt5" / "Qt5"
    resources = root / "resources"
    if resources.is_dir():
        for src in resources.iterdir():
            if not src.is_file():
                continue
            dest = exe_dir / src.name
            if dest.exists():
                continue
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
    locales = root / "translations" / "qtwebengine_locales"
    link = exe_dir / "qtwebengine_locales"
    if locales.is_dir() and not link.exists():
        flags = 0x08000000
        subprocess.run(
            ["cmd", "/c", f'mklink /J "{link}" "{locales}"'],
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if not link.exists():
            shutil.copytree(locales, link)


_prepare_webengine()

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

_utf8_codec = QTextCodec.codecForName(b"UTF-8")
if _utf8_codec is not None:
    QTextCodec.setCodecForLocale(_utf8_codec)

import score_db

QWebEnginePage = None
QWebEngineSettings = None
QWebEngineView = None

PICK_URL = "http://127.0.0.1:1429/picked"
_picked = {"body": b"", "mime": "application/octet-stream", "name": ""}
_pick_server_started = False


_ocr_sessions = {}
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_JPEDITOR_DIR = _BUNDLE_DIR / "jpeditor"
_JPEDITOR_DIST = _JPEDITOR_DIR / "dist"
OCR_DIR = _JPEDITOR_DIST / "redist" / "ocr"
if not OCR_DIR.is_dir():
    OCR_DIR = _JPEDITOR_DIR / "public" / "redist" / "ocr"


def _ocr_session(model):
    sess = _ocr_sessions.get(model)
    if sess is not None:
        return sess
    import onnxruntime as ort
    name = "ch_PP-OCRv4_det_infer.onnx" if model == "det" else "ch_PP-OCRv6_small_rec_infer.onnx"
    # 用内存加载，避免安装目录里的中文路径传到 onnxruntime 的窄字符接口。
    sess = ort.InferenceSession((OCR_DIR / name).read_bytes(), providers=["CPUExecutionProvider"])
    _ocr_sessions[model] = sess
    return sess


def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-Model, X-Dims")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Expose-Headers", "X-Dims")


def ensure_pick_server():
    global _pick_server_started
    if _pick_server_started:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            _cors(self)
            self.end_headers()

        def do_GET(self):
            body = _picked["body"]
            mime = _picked["mime"]
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            _cors(self)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/infer":
                self.send_error(404)
                return
            try:
                import numpy as np
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                model = self.headers.get("X-Model", "rec")
                dims = [int(x) for x in (self.headers.get("X-Dims") or "").split(",") if x]
                arr = np.array(np.frombuffer(raw, dtype=np.float32).reshape(dims), dtype=np.float32, copy=True)
                sess = _ocr_session(model)
                out = np.ascontiguousarray(sess.run(None, {"x": arr})[0], dtype=np.float32)
                body = out.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Dims", ",".join(str(int(d)) for d in out.shape))
                _cors(self)
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                msg = str(exc).encode("utf-8", "replace")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                _cors(self)
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, fmt, *args):
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 1429), Handler)
    except OSError:
        _pick_server_started = True
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _pick_server_started = True


def _load_webengine():
    """在 QApplication 创建之后再加载内嵌浏览器。

    更早加载时，Qt 会按当时的当前目录缓存资源路径。打包目录名是中文时，
    这条路径会被写坏，之后点击“打开谱面”主程序会直接退出。
    """
    global QWebEnginePage, QWebEngineSettings, QWebEngineView, ScorePage, ScoreView
    if QWebEngineView is not None:
        return True
    try:
        from PyQt5.QtWebEngineWidgets import (
            QWebEnginePage as _Page,
            QWebEngineSettings as _Settings,
            QWebEngineView as _View,
        )
    except Exception:
        return False
    QWebEnginePage = _Page
    QWebEngineSettings = _Settings
    QWebEngineView = _View

    class ScorePage(QWebEnginePage):
        def javaScriptPrompt(self, origin, msg, default_value):
            try:
                return self._pick_file(msg)
            except Exception:
                return True, ""

        def _pick_file(self, msg):
            text = str(msg or "")
            if not text.startswith("PICK:"):
                return False, ""
            kind = text.split(":", 1)[1]
            if kind == "image":
                caption = "选择简谱图片"
                filt = "图片与 PDF (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.pdf)"
            else:
                caption = "选择乐谱"
                filt = "乐谱 (*.jpwabc *.123 *.xml *.musicxml *.abc *.pu *.txt *.png *.jpg *.jpeg *.pdf)"
            parent = self.view().window() if self.view() else None
            ensure_pick_server()
            # 系统文件框和内嵌浏览器抢同一套窗口资源时，打包后的程序会直接退出。
            path, _ = QFileDialog.getOpenFileName(
                parent, caption, "", filt, options=QFileDialog.DontUseNativeDialog
            )
            if not path:
                return True, ""
            try:
                with open(path, "rb") as fh:
                    _picked["body"] = fh.read()
            except OSError:
                return True, ""
            _picked["name"] = os.path.basename(path)
            _picked["mime"] = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return True, "ok\n" + _picked["name"]

    class ScoreView(QWebEngineView):
        def __init__(self):
            super().__init__()
            self.setPage(ScorePage(self))
            # QStackedWidget hides this view while the dev server is starting.  On
            # Qt WebEngine that can leave the previous sibling's compositor tiles
            # visible when the view is shown again (the exact "old page behind the
            # new page" artefact seen in the embedded editor).  Make the web view
            # an opaque paint surface so every expose repaints its complete area.
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
            self.setAutoFillBackground(True)
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor("#0f0f0f"))
            palette.setColor(QPalette.Base, QColor("#0f0f0f"))
            self.setPalette(palette)
            settings = self.settings()
            settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
            settings.setAttribute(QWebEngineSettings.ScrollAnimatorEnabled, True)
            settings.setAttribute(QWebEngineSettings.FullScreenSupportEnabled, True)
            settings.setAttribute(QWebEngineSettings.ErrorPageEnabled, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            self.setZoomFactor(0.85)

    return True


JPEDITOR_DIR = _JPEDITOR_DIR if _JPEDITOR_DIR.is_dir() else score_db.DB_PATH.parent.parent / "jpeditor"
JPEDITOR_DIST = _JPEDITOR_DIST if _JPEDITOR_DIST.is_dir() else JPEDITOR_DIR / "dist"
JPEDITOR_URL = "http://127.0.0.1:1420"
NPM_CANDIDATES = [
    r"C:\app\nodejs\npm.cmd",
    r"C:\Program Files\nodejs\npm.cmd",
]

# ==================== 基础映射 ====================
BASE_FREQ = {
    '1': 262, '2': 294, '3': 330, '4': 349,
    '5': 393, '6': 440, '7': 494, '1·': 524,
}

KEY_MAP = {
    'Z': '1', 'X': '2', 'C': '3', 'V': '4',
    'B': '5', 'N': '6', 'M': '7', ',': '1·',
}

KEY_ORDER = ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',']
NOTE_TO_KEY = {v: k for k, v in KEY_MAP.items() if k != ','}
NOTE_TO_KEY['1'] = 'Z'

BINDING_MODES = ("none", "left", "middle", "right")
BINDING_LABELS = {
    "none": "正常音调",
    "left": "按住鼠标左键 · 降调",
    "middle": "按住鼠标中键 · 半音",
    "right": "按住鼠标右键 · 升调",
}
DEFAULT_BINDINGS = {
    mode: ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',']
    for mode in BINDING_MODES
}
PITCH_BINDINGS = {mode: list(keys) for mode, keys in DEFAULT_BINDINGS.items()}
DEFAULT_HOTKEYS = {
    "global_play": "F9",
    "global_stop": "F10",
    "previous_song": "F7",
    "next_song": "F8",
}


def binding_key(num, mouse=None):
    mode = mouse if mouse in PITCH_BINDINGS else "none"
    try:
        return PITCH_BINDINGS[mode][int(num) - 1]
    except (KeyError, IndexError, ValueError):
        return DEFAULT_BINDINGS["none"][int(num) - 1]


def apply_bindings(bindings):
    """更新全局键位映射，让解析、手动演奏和按键图谱使用同一套配置。"""
    for mode in BINDING_MODES:
        values = list(bindings.get(mode, DEFAULT_BINDINGS[mode]))
        if len(values) == 8:
            PITCH_BINDINGS[mode] = values
    KEY_MAP.clear()
    for index, key in enumerate(PITCH_BINDINGS["none"]):
        KEY_MAP[key] = str(index + 1) if index < 7 else "1·"
    KEY_ORDER[:] = list(PITCH_BINDINGS["none"])
    NOTE_TO_KEY.clear()
    for index, key in enumerate(KEY_ORDER[:7]):
        NOTE_TO_KEY[str(index + 1)] = key
    NOTE_TO_KEY["1"] = KEY_ORDER[0]


def key_to_vk(key):
    key = str(key or "").upper()
    if len(key) == 1 and "A" <= key <= "Z":
        return ord(key)
    if len(key) == 1 and "0" <= key <= "9":
        return ord(key)
    return {
        ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA,
        "'": 0xDE, "[": 0xDB, "]": 0xDD, "-": 0xBD,
        "=": 0xBB, "\\": 0xDC, "`": 0xC0,
    }.get(key)


class BeepPlayer:
    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="jianpu-preview")
        self._thread.start()

    def play(self, freq, duration_ms=260):
        request = (int(freq), max(int(duration_ms), 40))
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            pass

    def _run(self):
        import winsound
        while True:
            freq, duration_ms = self._queue.get()
            try:
                winsound.Beep(freq, duration_ms)
            except Exception as exc:
                print("Beep error:", exc)


class ScoreListWidget(QListWidget):
    """歌单里的歌曲可拖到其他歌单，也可在当前歌单内调整顺序。"""
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        # DragDrop allows both reordering here and dropping a song onto a
        # different歌单；dropEvent distinguishes the two cases.
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if not item:
            return
        mime = QMimeData()
        mime.setText(f"jianpu-score:{item.data(Qt.UserRole)}")
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.MoveAction)

    def dropEvent(self, event):
        if event.source() is self:
            super().dropEvent(event)
            self.owner.reorder_history()
            event.accept()
            return
        event.ignore()


class GroupListWidget(QListWidget):
    """接收歌曲拖放并交给主窗口完成数据库移动。"""
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().text().startswith("jianpu-score:"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        text = event.mimeData().text()
        if not text.startswith("jianpu-score:"):
            event.ignore()
            return
        target = self.itemAt(event.pos())
        if not target:
            event.ignore()
            return
        try:
            score_id = int(text.split(":", 1)[1])
            group_id = int(target.data(Qt.UserRole))
        except (TypeError, ValueError):
            event.ignore()
            return
        self.owner.move_score_to_group(score_id, group_id)
        event.acceptProposedAction()


# 全局按键：只把键鼠送给前台窗口（游戏口风琴），不播放系统声音。
_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
_user32 = ctypes.windll.user32

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MOUSE_DOWN = {"left": 0x0002, "middle": 0x0020, "right": 0x0008}
_MOUSE_UP = {"left": 0x0004, "middle": 0x0040, "right": 0x0010}
_VK = {
    "Z": 0x5A, "X": 0x58, "C": 0x43, "V": 0x56,
    "B": 0x42, "N": 0x4E, "M": 0x4D, ",": 0xBC,
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


def _send_input(item):
    _user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(item))


def _tap_key(vk, up=False):
    scan = _user32.MapVirtualKeyW(vk, 0)
    item = _INPUT(type=_INPUT_KEYBOARD)
    flags = _KEYEVENTF_SCANCODE | (_KEYEVENTF_KEYUP if up else 0)
    item.u.ki = _KEYBDINPUT(0, scan, flags, 0, 0)
    _send_input(item)


def _tap_mouse(flag):
    item = _INPUT(type=_INPUT_MOUSE)
    item.u.mi = _MOUSEINPUT(0, 0, 0, flag, 0, 0)
    _send_input(item)


class GlobalKeys:
    def __init__(self):
        self.vk = None
        self.mouse = None

    def hold(self, key, mouse):
        self.release()
        vk = key_to_vk(key)
        if not vk:
            return
        if mouse in _MOUSE_DOWN:
            _tap_mouse(_MOUSE_DOWN[mouse])
            self.mouse = mouse
        _tap_key(vk, up=False)
        self.vk = vk

    def release(self):
        if self.vk:
            _tap_key(self.vk, up=True)
            self.vk = None
        if self.mouse in _MOUSE_UP:
            _tap_mouse(_MOUSE_UP[self.mouse])
            self.mouse = None


class KeyChartWidget(QWidget):
    """面向玩家的按键图谱：显示按键、鼠标修饰和简谱音符。"""

    COLORS = {
        None: ("#252d39", "#e8edf5", "正常"),
        "left": ("#17364a", "#78c8ff", "左键·低音"),
        "middle": ("#352752", "#c9a8ff", "中键·半音"),
        "right": ("#4a3215", "#ffc15a", "右键·高音"),
        "rest": ("#20252d", "#7f8998", "休止"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.meta = {}
        self.notes = []
        self.active_index = -1
        self.scroll_area = None
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

    def set_score(self, meta, notes):
        self.meta = dict(meta or {})
        self.notes = list(notes or [])
        self.active_index = -1
        self._sync_height()
        self.update()

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def set_active(self, index):
        self.active_index = index
        self.update()
        if self.scroll_area is not None and 0 <= index < len(self.notes):
            QTimer.singleShot(0, self._scroll_to_active)

    def _scroll_to_active(self):
        if self.scroll_area is None or not (0 <= self.active_index < len(self.notes)):
            return
        width = max(self.width(), 720)
        margin, gap, tile_w, tile_h, rows = self._layout(width)[:5]
        for row, row_indices in enumerate(rows):
            if self.active_index in row_indices:
                col = row_indices.index(self.active_index)
                x = margin + col * (tile_w + gap)
                y = 86 + row * (tile_h + gap)
                viewport_h = self.scroll_area.viewport().height()
                bar = self.scroll_area.verticalScrollBar()
                bar.setValue(max(0, y - max(0, viewport_h // 2 - tile_h)))
                self.scroll_area.ensureVisible(x, y, 30, 110)
                return

    def _rows(self, width):
        margin, gap, tile_w, tile_h = 18, 10, 86, 72
        usable = max(tile_w, width - margin * 2)
        cols = max(1, int((usable + gap) / (tile_w + gap)))
        rows = []
        current = []
        current_bar = None
        for index, note in enumerate(self.notes):
            bar = note.get("bar")
            # 每个小节单独成行；没有小节标记的旧谱按可读宽度自动换行。
            new_phrase = current and bar is not None and current_bar is not None and bar != current_bar
            if new_phrase or len(current) >= cols:
                rows.append(current)
                current = []
            current.append(index)
            current_bar = bar
        if current:
            rows.append(current)
        return margin, gap, tile_w, tile_h, rows

    def _layout(self, width):
        margin, gap, tile_w, tile_h, rows = self._rows(width)
        total_h = 86 + max(1, len(rows)) * (tile_h + gap) + 18
        return margin, gap, tile_w, tile_h, rows, total_h

    def _sync_height(self):
        width = max(self.width(), 720)
        self.setMinimumHeight(self._layout(width)[-1])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_chart(painter, self.width(), self.height())

    def _paint_chart(self, painter, width, height):
        painter.fillRect(0, 0, width, height, QColor("#10141a"))
        title = self.meta.get("title") or "按键图谱"
        detail = "{}  ·  {}  ·  ♩={}".format(
            self.meta.get("key") or "1=C",
            self.meta.get("meter") or "4/4",
            self.meta.get("bpm") or 100,
        )
        painter.setPen(QColor("#f4f6fb"))
        painter.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        painter.drawText(18, 30, title)
        painter.setPen(QColor("#8b93a1"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(18, 52, detail)
        painter.setPen(QColor("#667080"))
        painter.drawText(18, 72, "按顺序演奏 · 颜色代表需要按住的鼠标键")

        margin, gap, tile_w, tile_h, rows, _ = self._layout(width)
        start_y = 86
        for row, row_indices in enumerate(rows):
            if row_indices and self.notes[row_indices[0]].get("bar") is not None:
                painter.setPen(QColor("#566070"))
                painter.setFont(QFont("Microsoft YaHei", 8))
                painter.drawText(18, start_y + row * (tile_h + gap) - 5,
                                 f"第 {self.notes[row_indices[0]].get('bar', 0) + 1} 小节")
            for col, index in enumerate(row_indices):
                note = self.notes[index]
                x = margin + col * (tile_w + gap)
                y = start_y + row * (tile_h + gap)
                is_rest = bool(note.get("is_rest") or not note.get("key"))
                kind = "rest" if is_rest else note.get("mouse")
                bg, fg, label = self.COLORS.get(kind, self.COLORS[None])
                rect = QRectF(x, y, tile_w, tile_h)
                if index == self.active_index:
                    painter.setPen(QPen(QColor("#f5a524"), 3))
                    painter.setBrush(QColor("#3d321c"))
                else:
                    painter.setPen(QPen(QColor("#343c4a"), 1))
                    painter.setBrush(QColor(bg))
                painter.drawRoundedRect(rect, 9, 9)

                painter.setPen(QColor("#768091"))
                painter.setFont(QFont("Microsoft YaHei", 7))
                painter.drawText(QRectF(x + 7, y + 5, 24, 12), Qt.AlignLeft, str(index + 1))
                painter.setPen(QColor(fg))
                painter.setFont(QFont("Consolas", 18, QFont.Bold))
                key_text = "—" if is_rest else str(note.get("key") or "—")
                painter.drawText(QRectF(x, y + 13, tile_w, 30), Qt.AlignCenter, key_text)
                painter.setFont(QFont("Microsoft YaHei", 7))
                painter.drawText(QRectF(x + 4, y + 43, tile_w - 8, 13), Qt.AlignCenter, label)
                painter.setPen(QColor("#aab2bf"))
                painter.drawText(
                    QRectF(x + 4, y + 56, tile_w - 8, 12),
                    Qt.AlignCenter,
                    "0" if is_rest else str(note.get("desc") or ""),
                )

    def export_png(self, path):
        export_width = 1400
        export_height = self._layout(export_width)[-1]
        image = QImage(export_width, export_height, QImage.Format_ARGB32)
        image.fill(QColor("#10141a"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_chart(painter, export_width, export_height)
        painter.end()
        return image.save(path, "PNG")


class GlobalHotkeyFilter(QAbstractNativeEventFilter):
    WM_HOTKEY = 0x0312

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def nativeEventFilter(self, event_type, message):
        if event_type in ("windows_generic_MSG", "windows_dispatcher_MSG"):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == self.WM_HOTKEY:
                hotkey_id = int(msg.wParam)
                QTimer.singleShot(0, lambda i=hotkey_id: self.callback(i))
                return True, 0
        return False, 0


class BindingDialog(QDialog):
    def __init__(self, bindings, hotkeys=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("按键设置")
        self.setMinimumWidth(720)
        self.setStyleSheet("QLineEdit { min-width: 48px; min-height: 30px; text-align: center; }")
        self.edits = {}
        self.hotkey_edits = {}
        layout = QVBoxLayout(self)
        intro = QLabel("这里设置游戏里的 8 个基础键。鼠标左键、中键、右键仍然分别负责降调、半音和升调。")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        grid.addWidget(QLabel("基础键位"), 0, 0)
        for col, note in enumerate(("1", "2", "3", "4", "5", "6", "7", "1·"), 1):
            label = QLabel(note)
            label.setAlignment(Qt.AlignCenter)
            grid.addWidget(label, 0, col)
        self.edits["none"] = []
        for col, value in enumerate(bindings.get("none", DEFAULT_BINDINGS["none"]), 1):
            edit = QLineEdit(str(value))
            edit.setMaxLength(1)
            edit.setAlignment(Qt.AlignCenter)
            self.edits["none"].append(edit)
            grid.addWidget(edit, 1, col)
        layout.addLayout(grid)
        tip = QLabel(
            "鼠标左键：降一个八度　·　鼠标中键：升半音　·　鼠标右键：升一个八度\n"
            "例如基础键位的 1 填 Z，那么 Z + 右键就是高音 1。\n"
            "下方可修改全局快捷键；Space 预览，Esc 停止。"
        )
        tip.setObjectName("muted")
        layout.addWidget(tip)
        hotkey_title = QLabel("播放快捷键")
        hotkey_title.setObjectName("section")
        layout.addWidget(hotkey_title)
        hotkey_grid = QGridLayout()
        hotkey_labels = {
            "global_play": "全局播放",
            "global_stop": "停止播放",
            "previous_song": "上一首",
            "next_song": "下一首",
        }
        saved_hotkeys = hotkeys or DEFAULT_HOTKEYS
        for row, name in enumerate(("global_play", "global_stop", "previous_song", "next_song")):
            hotkey_grid.addWidget(QLabel(hotkey_labels[name]), row, 0)
            edit = QKeySequenceEdit(QKeySequence(saved_hotkeys.get(name, DEFAULT_HOTKEYS[name])))
            edit.setToolTip("点击后按下快捷键组合")
            self.hotkey_edits[name] = edit
            hotkey_grid.addWidget(edit, row, 1)
        layout.addLayout(hotkey_grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self):
        values = {}
        row = []
        for edit in self.edits["none"]:
            value = edit.text().strip()
            if len(value) != 1:
                QMessageBox.warning(self, "按键无效", "每个按键必须填写一个字符。")
                return
            row.append(value.upper() if value.isascii() and value.isalpha() else value)
        if len(set(row)) != len(row):
            QMessageBox.warning(self, "按键重复", "8 个基础键不能有重复按键。")
            return
        values = {mode: list(row) for mode in BINDING_MODES}
        self.values = values
        self.hotkeys = {
            name: edit.keySequence().toString(QKeySequence.NativeText) or DEFAULT_HOTKEYS[name]
            for name, edit in self.hotkey_edits.items()
        }
        if len(set(self.hotkeys.values())) != len(self.hotkeys):
            QMessageBox.warning(self, "快捷键重复", "四个播放快捷键不能重复。")
            return
        for value in self.hotkeys.values():
            if QKeySequence(value).count() != 1 or not self.parent()._native_hotkey(value)[1]:
                QMessageBox.warning(self, "快捷键无效", "请使用单个字母、数字、功能键或组合键。")
                return
        self.accept()


# ==================== 元数据解析（已修复Q识别） ====================
def parse_metadata(text: str):
    meta = {
        "key": "1=C",
        "meter": "4/4",
        "bpm": 100,
        "title": "未命名"
    }

    # 标题
    m = re.search(r'T[:：]\s*(.+)', text)
    if m:
        meta["title"] = m.group(1).strip().split('\n')[0]

    # 调号
    m = re.search(r'K[:：]\s*([^\n]+)', text)
    if m:
        meta["key"] = m.group(1).strip()

    # 拍号
    m = re.search(r'M[:：]\s*([^\n]+)', text)
    if m:
        meta["meter"] = m.group(1).strip()

    # ---------- 速度识别（重点修复） ----------
    # 支持：
    # Q:1/4=105
    # Q:1=176
    # Q:176
    # Q:♩=120
    # Q: 1 = 176
    m = re.search(
        r'Q[:：]\s*(?:[1♩]/?\s*/?\s*4?\s*=\s*)?(\d{2,3})',
        text,
        re.IGNORECASE
    )
    if m:
        bpm = int(m.group(1))
        if 40 <= bpm <= 240:
            meta["bpm"] = bpm

    return meta


# 音符 / 倚音 / 括号。括号是连音线：同音为延音（只响一次），异音为圆滑（分别奏）。
_TOKEN_RE = re.compile(
    r"\{(?P<grace>[#♯b♭=]*[·,]?[0-7][',·]*)\}"
    r"|(?P<paren>[()])"
    r"|(?P<note>[#♯b♭=]*[·,]?[0-7](?:'+|,+|·+)?(?:[_\-.]+)?)"
)
_NOTE_RE = re.compile(
    r"^(?P<acc>[#♯b♭=]*)(?P<pre>[·,]+)?(?P<num>[0-7])(?P<oct>'+|,+|·+)?(?P<suf>[_\-.]*)$"
)


def extract_score_body(text: str) -> str:
    """去掉标题、速度、歌词行，保留谱面。"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("%"):
            continue
        # T: K: M: Q: w: w1: 等，避免把 w1 里的数字当成音符
        if re.match(r"^[A-Za-z]+\d*\s*:", line):
            continue
        lines.append(line)
    body = "\n".join(lines)
    body = re.sub(r"!.*?!", " ", body)
    return body


def duration_beats(suffix: str) -> float:
    """按符号从左到右计算以四分音符为单位的时值。

    1 四分，1_ 八分，1__ 十六分，1- 二分，1-- 三拍，1. 附点四分，1_. 附点八分。
    """
    beats = 1.0
    for ch in suffix:
        if ch == "_":
            beats *= 0.5
        elif ch == "-":
            beats += 1.0
        elif ch == ".":
            beats *= 1.5
    return beats


def beats_per_bar(meter: str) -> float:
    m = re.match(r"(\d+)\s*/\s*(\d+)", meter or "")
    if not m:
        return 4.0
    num, den = int(m.group(1)), int(m.group(2))
    if den <= 0:
        return 4.0
    return num * (4.0 / den)


def _modifier_label(octave: int, accidental: int) -> str:
    parts = []
    if octave >= 2:
        parts.append(f"升{octave}个八度")
    elif octave == 1:
        parts.append("升调")
    elif octave == -1:
        parts.append("降调")
    elif octave <= -2:
        parts.append(f"降{-octave}个八度")
    if accidental > 0:
        parts.append("升半音")
    elif accidental < 0:
        parts.append("降半音")
    return "+".join(parts) if parts else "正常"


def parse_note_token(token: str, base_quarter_ms: float, is_grace: bool = False):
    token = token.strip()
    m = _NOTE_RE.match(token)
    if not m:
        return None

    num = m.group("num")
    suffix = m.group("suf") or ""
    beats = duration_beats(suffix)

    if num == "0":
        return {
            "jianpu": token,
            "key": None,
            "mouse": None,
            "desc": "休止",
            "beats": beats,
            "duration_ms": base_quarter_ms * beats,
            "is_rest": True,
            "is_grace": False,
            "tied": False,
        }

    acc_s = m.group("acc") or ""
    if "#" in acc_s or "♯" in acc_s:
        accidental = 1
    elif "b" in acc_s or "♭" in acc_s:
        accidental = -1
    else:
        accidental = 0

    pre = m.group("pre") or ""
    oct_s = m.group("oct") or ""
    # jpeditor：' 高音，, 低音，. 是附点不是低八度。· 写在数字前为低音，写在后面为高音。
    octave = oct_s.count("'") + oct_s.count("·") - oct_s.count(",")
    octave -= pre.count("·") + pre.count(",")

    acc_mark = "#" if accidental > 0 else ("b" if accidental < 0 else "")
    if octave > 0:
        oct_mark = "'" * octave
    elif octave < 0:
        oct_mark = "," * (-octave)
    else:
        oct_mark = ""
    desc = f"{acc_mark}{num}{oct_mark}"
    if is_grace:
        desc = "倚" + desc

    if octave > 0:
        mouse = "right"
    elif octave < 0:
        mouse = "left"
    elif accidental:
        mouse = "middle"
    else:
        mouse = None

    return {
        "jianpu": token,
        "key": binding_key(num, mouse),
        "mouse": mouse,
        "desc": desc,
        "mod_label": _modifier_label(octave, accidental),
        "freq_base": num,
        "num": num,
        "octave": octave,
        "accidental": accidental,
        "beats": beats,
        "duration_ms": base_quarter_ms * beats,
        "is_rest": False,
        "is_grace": is_grace,
        "tied": False,
    }


def _same_pitch(a: dict, b: dict) -> bool:
    return (
        not a.get("is_rest")
        and not b.get("is_rest")
        and a.get("num") == b.get("num")
        and a.get("octave") == b.get("octave")
        and a.get("accidental") == b.get("accidental")
    )


def _absorb_grace(notes, grace, note, base_quarter_ms):
    """倚音不占谱面时值，从后面的音上匀出很短的一截。"""
    steal = min(0.125, note["beats"] * 0.5)
    if steal < 0.05:
        steal = 0.125
    else:
        note["beats"] -= steal
        note["duration_ms"] = base_quarter_ms * note["beats"]
    grace["beats"] = steal
    grace["duration_ms"] = base_quarter_ms * steal
    notes.append(grace)


def convert_jianpu(text: str):
    meta = parse_metadata(text)
    bpm = meta["bpm"]
    base_quarter_ms = 60000.0 / bpm
    body = extract_score_body(text)

    notes = []
    depth = 0
    tie_target = None
    pending_grace = None
    cursor = 0
    current_bar = 0

    for match in _TOKEN_RE.finditer(body):
        if "|" in body[cursor:match.start()]:
            current_bar += body[cursor:match.start()].count("|")
        cursor = match.end()
        if match.group("paren"):
            if match.group("paren") == "(":
                depth += 1
            else:
                depth = max(0, depth - 1)
                if depth == 0:
                    tie_target = None
            continue

        is_grace = match.group("grace") is not None
        raw = match.group("grace") if is_grace else match.group("note")
        note = parse_note_token(raw, base_quarter_ms, is_grace=is_grace)
        if not note:
            continue

        if is_grace:
            pending_grace = note
            continue

        note["bar"] = current_bar

        if pending_grace is not None:
            _absorb_grace(notes, pending_grace, note, base_quarter_ms)
            pending_grace = None

        if (
            depth > 0
            and tie_target is not None
            and _same_pitch(tie_target, note)
        ):
            tie_target["beats"] += note["beats"]
            tie_target["duration_ms"] = base_quarter_ms * tie_target["beats"]
            tie_target["jianpu"] = f"{tie_target['jianpu']}~{note['jianpu']}"
            tie_target["tied"] = True
        else:
            notes.append(note)
            tie_target = note if depth > 0 and not note["is_rest"] else None

    # 小节线不在音符正则里，单独扫一遍，避免把跨行的小节拆开
    # 上面的 bar_beats 会把全曲加在一起；这里按 | 重新切。
    bar_sums = []
    bar_beats = 0.0
    saw_bar_content = False
    cursor = 0
    for match in _TOKEN_RE.finditer(body):
        gap = body[cursor:match.start()]
        if "|" in gap and saw_bar_content:
            bar_sums.append(bar_beats)
            bar_beats = 0.0
            saw_bar_content = False
        cursor = match.end()
        if match.group("grace") or match.group("paren"):
            continue
        note = parse_note_token(match.group("note"), base_quarter_ms)
        if not note:
            continue
        bar_beats += note["beats"]
        saw_bar_content = True
    if saw_bar_content:
        bar_sums.append(bar_beats)

    expected = beats_per_bar(meta.get("meter"))
    bad_bars = [
        i + 1 for i, total in enumerate(bar_sums)
        if abs(total - expected) > 0.03
    ]
    meta["bar_count"] = len(bar_sums)
    meta["bad_bars"] = bad_bars
    meta["tie_count"] = sum(1 for n in notes if n.get("tied"))
    return meta, notes


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.player = BeepPlayer()
        self.global_keys = GlobalKeys()
        self.play_mode = "preview"
        self.meta = {}
        self.notes = []
        self.play_index = 0
        self.is_playing = False
        self.active_key = None
        self.editing_score_id = None
        self._title_override = None
        self._editing_source_snapshot = ""
        self._creating_new_song = False
        self._page_before_editor = 0
        self.binding_settings = self._load_bindings()
        self.hotkey_settings = self._load_hotkeys()
        apply_bindings(self.binding_settings)
        self.init_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def _load_app_icon(self):
        png = _BUNDLE_DIR / "static" / "logo.png"
        if not png.is_file():
            png = _BUNDLE_DIR / "static" / "口琴.png"
        ico = _BUNDLE_DIR / "static" / "app.ico"
        icon = QIcon()
        if png.is_file():
            icon.addPixmap(QPixmap(str(png)))
        if ico.is_file():
            icon.addFile(str(ico))
        return icon

    def init_ui(self):
        self.setWindowTitle("\u4e09\u89d2\u6d32\u53e3\u98ce\u7434")
        self.setWindowIcon(self._load_app_icon())
        self.setMinimumSize(1180, 760)
        self.resize(1280, 820)
        self.setStyleSheet("""
            QWidget { background-color: #0c0e12; color: #e6e8ee; font-family: 'Microsoft YaHei'; font-size: 13px; }
            QLabel { background: transparent; }
            QLabel#brand { font-size: 16px; font-weight: 700; color: #f4f6fb; }
            QLabel#sub { color: #8b93a1; font-size: 12px; }
            QLabel#section { color: #f5a524; font-size: 13px; font-weight: 700; }
            QLabel#muted { color: #8b93a1; font-size: 12px; }
            QFrame#card {
                background: #161b22; border: 1px solid #2a3140; border-radius: 10px;
            }
            QFrame#topbar { background: #0c0e12; border: none; }
            QFrame#sidebar { background: #11161d; border-right: 1px solid #252d39; }
            QFrame#playerBar { background: #171d26; border: 1px solid #303947; border-radius: 10px; }
            QPlainTextEdit, QListWidget, QComboBox {
                background: #10141a; color: #e6e8ee; border: 1px solid #2a3140;
                border-radius: 8px; padding: 6px; font-size: 13px;
            }
            QListWidget#playlistList { background: #11161d; border: none; padding: 2px; }
            QListWidget#playlistList::item { padding: 7px 10px; border-radius: 6px; color: #aeb6c4; }
            QListWidget#playlistList::item:selected { background: #3a2c14; color: #ffc14d; }
            QListWidget::item { padding: 8px 10px; border-radius: 6px; }
            QListWidget::item:selected { background: #3a2c14; color: #ffc14d; }
            QListWidget::item:hover { background: #222833; }
            QComboBox::drop-down { border: none; width: 22px; }
            QPushButton {
                background: #202733; color: #e6e8ee; border: 1px solid #364152;
                border-radius: 8px; padding: 8px 14px; min-height: 30px; font-size: 13px;
            }
            QPushButton:hover { background: #2c3442; }
            QPushButton:pressed { background: #1b212b; }
            QPushButton[active="true"] { color: #f5a524; background: #2b2416; border-color: #6b4a19; }
            QPushButton[active="false"] { color: #aeb6c4; }
            QPushButton#sideNav { text-align: left; padding: 8px 12px; color: #9ca6b5; border: none; }
            QPushButton#sideNav:hover { background: #1d2631; color: #f4f6fb; }
            QPushButton#sideNav[active="true"] { background: #3a2c14; color: #ffc14d; border: none; }
            QPushButton#primary {
                background: #f5a524; color: #1a1203; border: none; font-weight: 700; min-height: 36px;
            }
            QPushButton#primary:hover { background: #ffb83a; }
            QPushButton#primary:pressed { background: #d98e12; }
            QPushButton#danger { color: #e8c4c4; }
            QSlider::groove:horizontal { height: 4px; background: #2a3140; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #f5a524; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
            }
            QScrollBar:vertical { background: transparent; width: 8px; }
            QScrollBar::handle:vertical { background: #343c4a; border-radius: 4px; }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        central = QWidget()
        self.stack.addWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(190)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 18, 16, 18)
        sidebar_layout.setSpacing(8)
        side_brand = QLabel("琴  \u4e09\u89d2\u6d32\u53e3\u98ce\u7434")
        side_brand.setObjectName("brand")
        side_brand.setWordWrap(True)
        sidebar_layout.addWidget(side_brand)
        side_desc = QLabel("游戏口风琴工具")
        side_desc.setObjectName("sub")
        sidebar_layout.addWidget(side_desc)
        sidebar_layout.addSpacing(22)
        side_title = QLabel("工作区")
        side_title.setObjectName("muted")
        sidebar_layout.addWidget(side_title)
        self.side_nav_home = QPushButton("⌂  首页")
        self.side_nav_library = QPushButton("♫  歌单")
        self.side_nav_create = QPushButton("✎  创作")
        self.side_nav_player = QPushButton("▶  播放器")
        self.side_nav_editor = QPushButton("▣  谱面")
        for button in (self.side_nav_home, self.side_nav_library, self.side_nav_create, self.side_nav_player, self.side_nav_editor):
            button.setObjectName("sideNav")
            button.setMinimumHeight(38)
            button.setFlat(True)
            sidebar_layout.addWidget(button)
        self.side_nav_home.clicked.connect(lambda: self.set_main_page(0))
        self.side_nav_library.clicked.connect(lambda: self.set_main_page(1))
        self.side_nav_create.clicked.connect(lambda: self.set_main_page(2))
        self.side_nav_player.clicked.connect(lambda: self.set_main_page(3))
        self.side_nav_editor.clicked.connect(self.open_score_page)
        sidebar_layout.addStretch()
        self.side_tip = QLabel()
        self.side_tip.setObjectName("muted")
        sidebar_layout.addWidget(self.side_tip)
        self._refresh_hotkey_help()
        layout.addWidget(sidebar)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 12, 18, 14)
        right_layout.setSpacing(12)
        layout.addWidget(right_panel, 1)

        topbar = QFrame()
        topbar.setObjectName("topbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(4, 0, 4, 0)
        mark = QLabel()
        logo = _BUNDLE_DIR / "static" / "logo.png"
        if not logo.is_file():
            logo = _BUNDLE_DIR / "static" / "口琴.png"
        mark.setPixmap(QPixmap(str(logo)).scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(28, 28)
        mark.setStyleSheet("background:transparent; border:none;")
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("\u4e09\u89d2\u6d32\u53e3\u98ce\u7434")
        brand.setObjectName("brand")
        sub = QLabel("选一首歌试听，或打开谱面做成新曲子")
        sub.setObjectName("sub")
        brand_box.addWidget(brand)
        brand_box.addWidget(sub)
        top.addWidget(mark)
        top.addSpacing(8)
        top.addLayout(brand_box)
        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.nav_home = QPushButton("首页")
        self.nav_library = QPushButton("歌单")
        self.nav_player = QPushButton("播放器")
        self.nav_editor = QPushButton("谱面")
        for button in (self.nav_home, self.nav_library, self.nav_player, self.nav_editor):
            button.setFlat(True)
            button.hide()
            nav.addWidget(button)
        self.nav_home.clicked.connect(lambda: self.set_main_page(0))
        self.nav_library.clicked.connect(lambda: self.set_main_page(1))
        self.nav_player.clicked.connect(lambda: self.set_main_page(3))
        self.nav_editor.clicked.connect(self.open_score_page)
        top.addSpacing(30)
        top.addLayout(nav)
        top.addStretch()
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(18)
        self.lbl_mod = self._make_status("调式", "正常")
        self.lbl_note = self._make_status("音符", "—")
        self.lbl_freq = self._make_status("频率", "—")
        self.lbl_key = self._make_status("按键", "—")
        for w in (self.lbl_mod, self.lbl_note, self.lbl_freq, self.lbl_key):
            status_layout.addWidget(w)
        self.btn_settings = QPushButton("按键设置")
        self.btn_settings.clicked.connect(self.open_binding_settings)
        top.addWidget(self.btn_settings)
        top.addWidget(status_frame)
        right_layout.addWidget(topbar)

        body = QHBoxLayout()
        body.setSpacing(12)

        side = QWidget()
        side.setMinimumWidth(420)
        side.setMaximumWidth(1600)
        side.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        side_layout.addWidget(self._section("曲库"))
        lib = self._card()
        library_splitter = QSplitter(Qt.Horizontal)
        outer_lib_layout = QVBoxLayout(lib)
        transfers = QHBoxLayout()
        for text, callback in (("导入歌单", self.import_playlists), ("导出所选歌单", self.export_selected_playlists), ("备份全部歌单", self.export_all_playlists)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            transfers.addWidget(button)
        transfers.addStretch()
        outer_lib_layout.addLayout(transfers)
        outer_lib_layout.addWidget(library_splitter, 1)
        group_panel = QWidget()
        group_layout = QVBoxLayout(group_panel)
        group_layout.setContentsMargins(0, 0, 0, 0)
        song_panel = QWidget()
        lib_layout = QVBoxLayout(song_panel)
        library_splitter.addWidget(group_panel)
        library_splitter.addWidget(song_panel)
        library_splitter.setStretchFactor(0, 0)
        library_splitter.setStretchFactor(1, 1)
        library_splitter.setSizes([300, 650])
        library_splitter.setChildrenCollapsible(False)
        lib_layout.setContentsMargins(12, 12, 12, 12)
        lib_layout.setSpacing(8)
        group_row = QHBoxLayout()
        group_title = QLabel("歌单")
        group_title.setObjectName("muted")
        group_row.addWidget(group_title)
        group_row.addStretch()
        self.btn_group_new = QPushButton("新建")
        self.btn_group_new.clicked.connect(self.create_group)
        self.btn_group_del = QPushButton("删除所选")
        self.btn_group_del.setObjectName("danger")
        self.btn_group_del.clicked.connect(self.delete_group)
        self.btn_group_rename = QPushButton("改名")
        self.btn_group_rename.clicked.connect(self.rename_group)
        group_row.addWidget(self.btn_group_new)
        group_row.addWidget(self.btn_group_rename)
        group_row.addWidget(self.btn_group_del)
        group_layout.addLayout(group_row)
        self.group_list = GroupListWidget(self)
        self.group_list.setObjectName("playlistList")
        self.group_list.setMinimumWidth(260)
        self.group_list.currentRowChanged.connect(lambda _row: self._on_group_changed())
        group_layout.addWidget(self.group_list, 1)
        group_hint = QLabel("Ctrl / Shift 多选 · Ctrl+A 全选")
        group_hint.setObjectName("muted")
        group_layout.addWidget(group_hint)
        self.search_songs = QLineEdit()
        self.search_songs.setPlaceholderText("搜索歌曲…")
        self.search_songs.setClearButtonEnabled(True)
        self.search_songs.textChanged.connect(
            lambda _text: self._fill_songs(self._selected_score_id())
        )
        lib_layout.addWidget(self.search_songs)
        self.history_list = ScoreListWidget(self)
        self.history_list.currentItemChanged.connect(self._on_song_selected)
        self.history_list.itemDoubleClicked.connect(lambda _item: self.play_history())
        lib_layout.addWidget(self.history_list, 1)
        hint_lib = QLabel("双击试听 · Ctrl / Shift 多选 · Ctrl+A 全选 · 拖动当前歌曲可移动")
        hint_lib.setObjectName("muted")
        lib_layout.addWidget(hint_lib)
        self.btn_hist_play = QPushButton("播放")
        self.btn_hist_play.setObjectName("primary")
        self.btn_hist_play.clicked.connect(self.play_history)
        self.btn_hist_load = QPushButton("载入")
        self.btn_hist_load.clicked.connect(self.load_history)
        self.btn_hist_rename = QPushButton("改名")
        self.btn_hist_rename.clicked.connect(self.rename_history)
        self.btn_hist_fav = QPushButton("收藏")
        self.btn_hist_fav.clicked.connect(self.toggle_favorite)
        self.btn_hist_move = QPushButton("移到…")
        self.btn_hist_move.clicked.connect(self.move_history)
        self.btn_hist_del = QPushButton("删除所选")
        self.btn_hist_del.setObjectName("danger")
        self.btn_hist_del.clicked.connect(self.delete_history)
        play_row_lib = QHBoxLayout()
        play_row_lib.addWidget(self.btn_hist_play, 1)
        play_row_lib.addWidget(self.btn_hist_load, 1)
        lib_layout.addLayout(play_row_lib)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.btn_hist_rename, 1)
        edit_row.addWidget(self.btn_hist_fav, 1)
        edit_row.addWidget(self.btn_hist_move, 1)
        edit_row.addWidget(self.btn_hist_del, 1)
        lib_layout.addLayout(edit_row)
        side_layout.addWidget(lib, 1)

        make = self._card()
        make.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        make.setMaximumHeight(620)
        make_layout = QVBoxLayout(make)
        make_layout.setContentsMargins(12, 12, 12, 12)
        make_layout.setSpacing(8)
        make_layout.setAlignment(Qt.AlignTop)
        self.btn_open_score = QPushButton("打开谱面")
        self.btn_open_score.setObjectName("primary")
        self.btn_open_score.setMinimumHeight(36)
        self.btn_open_score.clicked.connect(self.open_score_page)
        make_layout.addWidget(self.btn_open_score)
        self.lbl_editor = QLabel("")
        self.lbl_editor.setObjectName("muted")
        self.lbl_editor.setWordWrap(True)
        self.lbl_editor.setMaximumHeight(28)
        self.lbl_editor.hide()
        make_layout.addWidget(self.lbl_editor)
        paste_tip = QLabel("或粘贴简谱文本")
        paste_tip.setObjectName("muted")
        make_layout.addWidget(paste_tip)
        self.editing_banner = QLabel("新歌曲：转换后会保存到当前歌单")
        self.editing_banner.setObjectName("muted")
        self.editing_banner.setStyleSheet("color:#f5a524; padding:4px 0;")
        make_layout.addWidget(self.editing_banner)
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("把 jpeditor 简谱贴在这里")
        self.text_input.setMinimumHeight(280)
        self.text_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        make_layout.addWidget(self.text_input)
        btn_row = QHBoxLayout()
        self.btn_convert = QPushButton("转换成曲子")
        self.btn_convert.setObjectName("primary")
        # clicked 会附带 checked=False；直接连接会把它错误地传给
        # do_convert(show_overwrite_prompt)，从而关闭覆盖提示。
        self.btn_convert.clicked.connect(lambda _checked=False: self.do_convert(True))
        self.btn_clear = QPushButton("新歌曲")
        self.btn_clear.clicked.connect(self.new_song)
        btn_row.addWidget(self.btn_convert, 1)
        btn_row.addWidget(self.btn_clear)
        make_layout.addLayout(btn_row)
        # 创作卡片在下面放入独立的“创作”页面，歌单页只负责管理歌曲。

        stage = QWidget()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(8)
        stage_head = QHBoxLayout()
        stage_head.addWidget(self._section("按键演奏谱"))
        stage_head.addStretch()
        self.btn_export_chart = QPushButton("导出按键图谱")
        self.btn_export_chart.clicked.connect(self.export_key_chart)
        stage_head.addWidget(self.btn_export_chart)
        stage_layout.addLayout(stage_head)
        stage_card = self._card()
        stage_box = QVBoxLayout(stage_card)
        stage_box.setContentsMargins(14, 12, 14, 12)
        stage_box.setSpacing(10)
        self.lbl_meta = QLabel("还没有曲子。从左边曲库选一首，或打开谱面做一首。")
        self.lbl_meta.setObjectName("muted")
        self.lbl_meta.setWordWrap(True)
        stage_box.addWidget(self.lbl_meta)
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setFrameShape(QFrame.NoFrame)
        self.chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.key_chart = KeyChartWidget()
        self.key_chart.set_scroll_area(self.chart_scroll)
        self.chart_scroll.setWidget(self.key_chart)
        stage_box.addWidget(self.chart_scroll, 1)

        play_row = QHBoxLayout()
        self.btn_play = QPushButton("预览播放")
        self.btn_play.setObjectName("primary")
        self.btn_play.setMinimumHeight(36)
        self.btn_play.clicked.connect(lambda: self.start_sequence("preview"))
        self.btn_global = QPushButton("全局播放")
        self.btn_global.setMinimumHeight(36)
        self.btn_global.clicked.connect(self.start_global_countdown)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_sequence)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(60, 140)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(140)
        self.lbl_speed = QLabel("速度 100%")
        self.lbl_speed.setObjectName("muted")
        self.speed_slider.valueChanged.connect(lambda v: self.lbl_speed.setText(f"速度 {v}%"))
        self.lbl_progress = QLabel("准备就绪")
        self.lbl_progress.setObjectName("muted")
        play_row.addWidget(self.btn_play)
        play_row.addWidget(self.btn_global)
        play_row.addWidget(self.btn_stop)
        play_row.addSpacing(8)
        play_row.addWidget(self.speed_slider)
        play_row.addWidget(self.lbl_speed)
        play_row.addStretch()
        play_row.addWidget(self.lbl_progress)
        stage_box.addLayout(play_row)

        self.big_hint = QLabel("选一首歌，或先转换成曲子")
        self.big_hint.setAlignment(Qt.AlignCenter)
        self.big_hint.setFixedHeight(36)
        self.big_hint.setStyleSheet(self._hint_style("idle"))
        stage_box.addWidget(self.big_hint)

        key_frame = QFrame()
        key_frame.setStyleSheet("background:#10141a; border:1px solid #2a3140; border-radius:8px;")
        key_layout = QVBoxLayout(key_frame)
        key_layout.setContentsMargins(12, 8, 12, 10)
        tip = QLabel(
            "正常  Z1 X2 C3 V4 B5 N6 M7 ,高音1\n"
            "按住左键降八度 · 中键升半音 · 右键再高一个八度。全局播放只按这些键，不发声，请先切到游戏。"
        )
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        key_layout.addWidget(tip)
        self.key_buttons = {}
        row = QHBoxLayout()
        row.setSpacing(8)
        self.key_button_layout = row
        for k in KEY_ORDER:
            btn = QPushButton(f"{k}\n{KEY_MAP[k]}")
            btn.setFixedHeight(64)
            btn.setEnabled(False)
            btn.setStyleSheet(self._key_style(False))
            self.key_buttons[k] = btn
            row.addWidget(btn, 1)
        key_layout.addLayout(row)
        stage_box.addWidget(key_frame)
        stage_layout.addWidget(stage_card, 1)
        self.main_pages = QStackedWidget()
        self.home_page = self._build_home_page()
        self.playlist_page = QWidget()
        playlist_layout = QVBoxLayout(self.playlist_page)
        playlist_layout.setContentsMargins(0, 0, 0, 0)
        playlist_layout.addWidget(side, 1)
        self.create_page = QWidget()
        create_layout = QVBoxLayout(self.create_page)
        create_layout.setContentsMargins(0, 0, 0, 0)
        create_title = QLabel("创作新歌曲")
        create_title.setObjectName("section")
        create_layout.addWidget(create_title)
        create_body = QHBoxLayout()
        create_body.setSpacing(12)
        create_body.addWidget(make, 2)
        guide = self._card()
        guide.setMinimumWidth(260)
        guide_layout = QVBoxLayout(guide)
        guide_layout.setContentsMargins(20, 20, 20, 20)
        guide_title = QLabel("创作流程")
        guide_title.setObjectName("brand")
        guide_title.setStyleSheet("font-size:18px;")
        guide_layout.addWidget(guide_title)
        guide_text = QLabel(
            "1  打开谱面或粘贴简谱\n\n"
            "2  转换成歌曲，自动保存到当前歌单\n\n"
            "3  去播放器检查按键和节奏\n\n"
            "4  切到游戏后使用全局播放"
        )
        guide_text.setObjectName("muted")
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_text)
        guide_layout.addStretch()
        go_library = QPushButton("返回歌单")
        go_library.clicked.connect(lambda: self.set_main_page(1))
        guide_layout.addWidget(go_library)
        create_body.addWidget(guide, 1)
        create_layout.addLayout(create_body, 1)
        create_layout.addStretch()
        self.player_page = QWidget()
        player_layout = QVBoxLayout(self.player_page)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.addWidget(stage)
        self.main_pages.addWidget(self.home_page)
        self.main_pages.addWidget(self.playlist_page)
        self.main_pages.addWidget(self.create_page)
        self.main_pages.addWidget(self.player_page)
        right_layout.addWidget(self.main_pages, 1)
        self.set_main_page(0)
        self.player_bar = self._build_player_bar()
        right_layout.addWidget(self.player_bar)

        self.editor_page = QWidget()
        editor_page_layout = QVBoxLayout(self.editor_page)
        self.editor_page_layout = editor_page_layout
        editor_page_layout.setContentsMargins(4, 4, 4, 4)
        editor_page_layout.setSpacing(4)
        editor_bar = QHBoxLayout()
        editor_mark = QLabel("谱")
        editor_mark.setAlignment(Qt.AlignCenter)
        editor_mark.setFixedSize(28, 28)
        editor_mark.setStyleSheet("background:transparent; border:none;")
        editor_title = QLabel("谱面编辑")
        editor_title.setObjectName("brand")
        self.btn_close_score = QPushButton("关闭")
        self.btn_close_score.setMinimumSize(88, 34)
        self.btn_close_score.clicked.connect(self.close_score_page)
        self.btn_editor_fit = QPushButton("适应窗口")
        self.btn_editor_fit.clicked.connect(self._fit_editor_to_window)
        self.btn_editor_zoom_out = QPushButton("−")
        self.btn_editor_zoom_out.setFixedWidth(38)
        self.btn_editor_zoom_out.clicked.connect(lambda: self._change_editor_zoom(-0.1))
        self.btn_editor_zoom_reset = QPushButton("100%")
        self.btn_editor_zoom_reset.clicked.connect(lambda: self._set_editor_zoom(1.0))
        self.btn_editor_zoom_in = QPushButton("＋")
        self.btn_editor_zoom_in.setFixedWidth(38)
        self.btn_editor_zoom_in.clicked.connect(lambda: self._change_editor_zoom(0.1))
        editor_bar.addWidget(editor_mark)
        editor_bar.addSpacing(8)
        editor_bar.addWidget(editor_title)
        editor_bar.addStretch()
        editor_bar.addWidget(self.btn_editor_fit)
        editor_bar.addWidget(self.btn_editor_zoom_out)
        editor_bar.addWidget(self.btn_editor_zoom_reset)
        editor_bar.addWidget(self.btn_editor_zoom_in)
        editor_bar.addWidget(self.btn_close_score)
        editor_page_layout.addLayout(editor_bar)
        self.editor_view = None
        if QWebEngineView is not None:
            self.editor_placeholder = QLabel("打开谱面时加载编辑器…")
        else:
            self.editor_placeholder = QLabel("内嵌页面需要 PyQtWebEngine。请用 E:\\conda_env\\ai\\python.exe 启动本程序。")
        self.editor_placeholder.setWordWrap(True)
        editor_page_layout.addWidget(self.editor_placeholder, 1)
        self.stack.addWidget(self.editor_page)


        self.play_timer = QTimer(self)
        self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self._play_next_note)

        self._jpeditor_proc = None
        self._jpeditor_server = None
        self._editor_url_loaded = False
        self._editor_retries = 0
        self._server_retries = 0
        self._pending_editor_text = ""
        self._loading_groups = False
        self._countdown_left = 0
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._countdown_tick)
        self._install_shortcuts()
        self._register_global_hotkeys()
        self.refresh_history()

    def _ensure_editor_view(self):
        if self.editor_view is not None or QWebEngineView is None:
            return self.editor_view is not None
        if self.editor_placeholder is not None:
            self.editor_page_layout.removeWidget(self.editor_placeholder)
            self.editor_placeholder.deleteLater()
            self.editor_placeholder = None
        self.editor_view = ScoreView()
        self.editor_view.loadFinished.connect(self._on_editor_loaded)
        self.editor_page_layout.addWidget(self.editor_view, 1)
        return True

    def _set_editor_zoom(self, factor):
        if self.editor_view is None:
            return
        factor = max(0.5, min(1.5, float(factor)))
        self.editor_view.setZoomFactor(factor)
        self.btn_editor_zoom_reset.setText(f"{int(round(factor * 100))}%")

    def _change_editor_zoom(self, delta):
        if self.editor_view is not None:
            self._set_editor_zoom(self.editor_view.zoomFactor() + delta)

    def _fit_editor_to_window(self):
        if self.editor_view is None:
            return
        viewport_width = max(1, self.editor_view.width())
        self._set_editor_zoom(max(0.68, min(1.0, viewport_width / 1180.0)))

    def _build_player_bar(self):
        bar = QFrame()
        bar.setObjectName("playerBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 14, 8)
        row.setSpacing(8)
        self.bar_song = QLabel("未选择歌曲")
        self.bar_song.setObjectName("brand")
        self.bar_song.setMinimumWidth(180)
        row.addWidget(self.bar_song)
        row.addStretch()
        self.bar_prev = QPushButton("上一首")
        self.bar_prev.clicked.connect(self.previous_song)
        self.bar_play = QPushButton("▶ 预览")
        self.bar_play.setObjectName("primary")
        self.bar_play.clicked.connect(lambda: self.start_sequence("preview"))
        self.bar_next = QPushButton("下一首")
        self.bar_next.clicked.connect(self.next_song)
        self.bar_global = QPushButton("全局播放")
        self.bar_global.clicked.connect(self.start_global_countdown)
        self.bar_mode = QLabel("预览模式")
        self.bar_mode.setObjectName("muted")
        for button in (self.bar_prev, self.bar_play, self.bar_next, self.bar_global):
            button.setMinimumHeight(32)
            row.addWidget(button)
        row.addSpacing(10)
        row.addWidget(self.bar_mode)
        return bar

    def _build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)
        hero = QFrame()
        hero.setObjectName("card")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(26, 24, 26, 24)
        title = QLabel("今天想演奏什么？")
        title.setObjectName("brand")
        title.setStyleSheet("font-size:26px; color:#f4f6fb;")
        subtitle = QLabel("管理你的歌单，先听一遍，再切到三角洲全局演奏。")
        subtitle.setObjectName("sub")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        actions = QHBoxLayout()
        start = QPushButton("打开歌单")
        start.setObjectName("primary")
        start.clicked.connect(lambda: self.set_main_page(1))
        new_song = QPushButton("创作新歌曲")
        new_song.clicked.connect(self._new_song_from_home)
        actions.addWidget(start)
        actions.addWidget(new_song)
        actions.addStretch()
        hero_layout.addLayout(actions)
        layout.addWidget(hero)

        stats = QFrame()
        stats.setObjectName("card")
        stats_layout = QHBoxLayout(stats)
        stats_layout.setContentsMargins(20, 16, 20, 16)
        self.home_stats = QLabel("歌单正在加载…")
        self.home_stats.setObjectName("muted")
        stats_layout.addWidget(self.home_stats)
        stats_layout.addStretch()
        settings = QPushButton("设置按键和快捷键")
        settings.clicked.connect(self.open_binding_settings)
        stats_layout.addWidget(settings)
        layout.addWidget(stats)
        tutorial = self._card()
        tutorial_layout = QVBoxLayout(tutorial)
        tutorial_layout.addWidget(self._section("新手入门 · 从第一次打开到开始演奏"))
        self.tutorial_text = QLabel()
        self.tutorial_text.setWordWrap(True)
        tutorial_layout.addWidget(self.tutorial_text)
        layout.addWidget(tutorial)
        self._refresh_hotkey_help()
        layout.addStretch()
        return page

    def _refresh_hotkey_help(self):
        h = self.hotkey_settings
        self.side_tip.setText(f"{h['previous_song']} 上一首\n{h['next_song']} 下一首\n{h['global_play']} 全局播放\n{h['global_stop']} 停止播放")
        if hasattr(self, "tutorial_text"):
            self.tutorial_text.setText(
                "① 第一次使用：先点击“按键设置”，把 1～7 和高音 1 设置成游戏里对应的琴键；同时确认全局播放、停止、上一首、下一首快捷键没有和其他软件冲突。\n\n"
                "② 添加文字简谱：进入“创作”，粘贴带有 T（歌名）、K（调号）、M（拍号）的简谱，点击“转换成曲子”，确认音符图谱后保存到当前歌单。\n\n"
                "③ 添加图片简谱：进入“谱面”，点击导入图片或 PDF。识别完成后请检查调号、拍号、连音和歌词，发现错误时先校对再保存；图片越清晰、倾斜越小，识别结果越稳定。\n\n"
                "④ 管理歌库：在“歌单”页面左侧选择歌单，右侧搜索歌曲。可以新建、改名、移动和收藏歌曲；Ctrl / Shift 多选后可批量删除。定期使用“备份全部歌单”保存 JSON 文件，换电脑时用“导入歌单”恢复。\n\n"
                "⑤ 试听和演奏：双击歌曲或点击“预览播放”检查速度，确认无误后点击“全局播放”。倒计时开始后切回游戏并打开口风琴；"
                f"{h['global_stop']} 随时停止，{h['previous_song']} / {h['next_song']} 切换歌曲。播放过程中不要关闭本程序。")

    def _new_song_from_home(self):
        self.set_main_page(2)
        self.new_song()

    def set_main_page(self, index):
        if hasattr(self, "main_pages"):
            self.main_pages.setCurrentIndex(index)
        if index == 2 and hasattr(self, "text_input") and not self.editing_score_id and not self._creating_new_song:
            source = self.text_input.toPlainText().strip()
            if source:
                row = score_db.find_score_by_source(source)
                if row:
                    self.editing_score_id = int(row["id"])
                    self._editing_source_snapshot = source
                    self.editing_banner.setText(
                        f"正在编辑已有歌曲：{row.get('title') or '未命名'} · 点击转换时会询问保存方式"
                    )
                    self.lbl_meta.setText(
                        f"正在编辑：{row.get('title') or '未命名'} · 修改后转换会先询问保存方式"
                    )
        if hasattr(self, "side_nav_home"):
            for button, button_index in (
                (self.side_nav_home, 0), (self.side_nav_library, 1),
                (self.side_nav_create, 2), (self.side_nav_player, 3)
            ):
                button.setProperty("active", button_index == index)
                button.style().unpolish(button)
                button.style().polish(button)
        if hasattr(self, "nav_home"):
            for button, button_index in (
                (self.nav_home, 0), (self.nav_library, 1), (self.nav_player, 2)
            ):
                button.setProperty("active", button_index == index)
                button.style().unpolish(button)
                button.style().polish(button)

    def _refresh_home_stats(self):
        if not hasattr(self, "home_stats"):
            return
        groups = score_db.list_groups()
        songs = score_db.list_scores()
        favorites = sum(1 for song in songs if song.get("is_favorite"))
        self.home_stats.setText(
            f"{len(groups)} 个歌单   ·   {len(songs)} 首歌曲   ·   {favorites} 首收藏"
        )

    def _load_bindings(self):
        try:
            raw = score_db.get_setting("keyboard_bindings")
            if raw:
                saved = json.loads(raw)
                return {
                    mode: list(saved.get(mode, DEFAULT_BINDINGS[mode]))
                    for mode in BINDING_MODES
                }
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return {mode: list(keys) for mode, keys in DEFAULT_BINDINGS.items()}

    def _load_hotkeys(self):
        try:
            raw = score_db.get_setting("playback_hotkeys")
            if raw:
                saved = json.loads(raw)
                return {name: saved.get(name, value) for name, value in DEFAULT_HOTKEYS.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return dict(DEFAULT_HOTKEYS)

    def open_binding_settings(self):
        dialog = BindingDialog(self.binding_settings, self.hotkey_settings, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self.binding_settings = dialog.values
        apply_bindings(self.binding_settings)
        score_db.set_setting(
            "keyboard_bindings",
            json.dumps(self.binding_settings, ensure_ascii=False),
        )
        self.hotkey_settings = dialog.hotkeys
        self._refresh_hotkey_help()
        score_db.set_setting(
            "playback_hotkeys",
            json.dumps(self.hotkey_settings, ensure_ascii=False),
        )
        self.shortcut_global.setKey(QKeySequence(self.hotkey_settings["global_play"]))
        self.shortcut_prev.setKey(QKeySequence(self.hotkey_settings["previous_song"]))
        self.shortcut_next.setKey(QKeySequence(self.hotkey_settings["next_song"]))
        self._register_global_hotkeys()
        # 已有谱面立即重新绑定，用户无需重新载入歌曲。
        if self.text_input.toPlainText().strip():
            self.do_convert(show_overwrite_prompt=False)
        self._refresh_key_buttons()

    def _refresh_key_buttons(self):
        for key, button in list(self.key_buttons.items()):
            button.deleteLater()
        self.key_buttons.clear()
        row_layout = self.key_button_layout
        while row_layout.count():
            item = row_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for key in KEY_ORDER:
            btn = QPushButton(f"{key}\n{KEY_MAP.get(key, '')}")
            btn.setFixedHeight(64)
            btn.setEnabled(False)
            btn.setStyleSheet(self._key_style(False))
            self.key_buttons[key] = btn
            row_layout.addWidget(btn, 1)

    def _section(self, text):
        lab = QLabel(text)
        lab.setObjectName("section")
        return lab

    def _card(self):
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def _hint_style(self, tone="idle"):
        border, fg = {
            "idle": ("#2a3140", "#c5cad3"),
            "live": ("#f5a524", "#f5a524"),
            "rest": ("#3a4150", "#8b93a1"),
        }.get(tone, ("#2a3140", "#c5cad3"))
        return (
            f"background:#12161c; border-radius:8px; border:1px solid {border};"
            f"font-size:14px; font-weight:600; color:{fg};"
        )

    def _make_status(self, title, value):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet("color:#8b93a1; font-size:11px;")
        t.setAlignment(Qt.AlignCenter)
        v = QLabel(value)
        v.setStyleSheet("color:#f4f6fb; font-size:14px; font-weight:700;")
        v.setAlignment(Qt.AlignCenter)
        l.addWidget(t)
        l.addWidget(v)
        w.value_label = v
        return w

    def _install_shortcuts(self):
        self.shortcut_preview = QShortcut(QKeySequence("Space"), self)
        self.shortcut_preview.setContext(Qt.ApplicationShortcut)
        self.shortcut_preview.activated.connect(lambda: self.start_sequence("preview"))
        self.shortcut_stop = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_stop.setContext(Qt.ApplicationShortcut)
        self.shortcut_stop.activated.connect(self.stop_sequence)
        self.shortcut_global = QShortcut(QKeySequence(self.hotkey_settings["global_play"]), self)
        self.shortcut_global.setContext(Qt.ApplicationShortcut)
        self.shortcut_global.activated.connect(lambda: self.start_sequence("global"))
        self.shortcut_prev = QShortcut(QKeySequence(self.hotkey_settings["previous_song"]), self)
        self.shortcut_prev.setContext(Qt.ApplicationShortcut)
        self.shortcut_prev.activated.connect(lambda: self.previous_song(force_global=True))
        self.shortcut_next = QShortcut(QKeySequence(self.hotkey_settings["next_song"]), self)
        self.shortcut_next.setContext(Qt.ApplicationShortcut)
        self.shortcut_next.activated.connect(lambda: self.next_song(force_global=True))

    def _register_global_hotkeys(self):
        if not hasattr(self, "_hotkey_filter"):
            self._hotkey_filter = GlobalHotkeyFilter(self._on_global_hotkey)
            QApplication.instance().installNativeEventFilter(self._hotkey_filter)
        QTimer.singleShot(100, self._register_hotkey_native)

    def _register_hotkey_native(self):
        try:
            hwnd = int(self.winId())
            for hotkey_id in (1, 2, 3, 4):
                _user32.UnregisterHotKey(hwnd, hotkey_id)
            for hotkey_id, name in enumerate(
                ("global_play", "global_stop", "previous_song", "next_song"), 1
            ):
                modifiers, vk = self._native_hotkey(self.hotkey_settings[name])
                if vk:
                    _user32.RegisterHotKey(hwnd, hotkey_id, modifiers, vk)
        except Exception:
            pass

    def _native_hotkey(self, text):
        sequence = QKeySequence(text)
        if sequence.isEmpty():
            return 0, 0
        value = int(sequence[0])
        modifiers = 0
        if value & int(Qt.ControlModifier):
            modifiers |= 0x0002
        if value & int(Qt.AltModifier):
            modifiers |= 0x0001
        if value & int(Qt.ShiftModifier):
            modifiers |= 0x0004
        if value & int(Qt.MetaModifier):
            modifiers |= 0x0008
        key = value & 0x01ffffff
        if int(Qt.Key_F1) <= key <= int(Qt.Key_F35):
            return modifiers, 0x70 + key - int(Qt.Key_F1)
        if 0x41 <= key <= 0x5A or 0x30 <= key <= 0x39:
            return modifiers, key
        return modifiers, {
            int(Qt.Key_Comma): 0xBC,
            int(Qt.Key_Period): 0xBE,
            int(Qt.Key_Slash): 0xBF,
            int(Qt.Key_Semicolon): 0xBA,
            int(Qt.Key_Space): 0x20,
        }.get(key, 0)

    def _on_global_hotkey(self, hotkey_id):
        if hotkey_id == 1:
            self.start_sequence("global")
        elif hotkey_id == 2:
            self.stop_sequence()
        elif hotkey_id == 3:
            self.previous_song(force_global=True)
        elif hotkey_id == 4:
            self.next_song(force_global=True)

    def start_global_countdown(self):
        if not self.notes:
            QMessageBox.information(self, "提示", "请先转换一首歌")
            return
        if self.is_playing:
            self.stop_sequence()
            return
        self._countdown_left = 3
        self.btn_global.setText("切到游戏… 3")
        self.lbl_progress.setText(f"3 秒后开始全局演奏 · {self.hotkey_settings['global_stop']} 可停止")
        self.countdown_timer.start(1000)

    def _countdown_tick(self):
        self._countdown_left -= 1
        if self._countdown_left <= 0:
            self.countdown_timer.stop()
            self.start_sequence("global")
            return
        self.btn_global.setText(f"切到游戏… {self._countdown_left}")

    def _on_song_selected(self, current, _previous):
        if not current:
            return
        row = score_db.get_score(int(current.data(Qt.UserRole)))
        if not row:
            return
        self.lbl_meta.setText(
            f"{row.get('title') or '未命名'}  ·  {row.get('key_sig') or '1=C'}  ·  "
            f"{row.get('meter') or '4/4'}  ·  ♩={row.get('bpm') or 100}"
        )
        if hasattr(self, "bar_song"):
            self.bar_song.setText(row.get("title") or "未命名")
        self.btn_hist_fav.setText("已收藏" if row.get("is_favorite") else "收藏")

    def export_key_chart(self):
        if not self.notes:
            QMessageBox.information(self, "提示", "请先转换一首歌")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出按键图谱", f"{self.meta.get('title') or '按键图谱'}.png",
            "PNG 图片 (*.png)"
        )
        if path and self.key_chart.export_png(path):
            self.lbl_progress.setText(f"图谱已保存：{os.path.basename(path)}")

    def _key_style(self, active=False):
        if active:
            return (
                "QPushButton { background:#f5a524; color:#1a1203; border:none;"
                " border-radius:8px; font-size:13px; font-weight:700; }"
            )
        return (
            "QPushButton { background:#232a36; color:#e6e8ee; border:1px solid #343c4a;"
            " border-radius:8px; font-size:13px; font-weight:700; }"
        )

    def current_group_id(self):
        item = self.group_list.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def _select_group(self, group_id):
        for index in range(self.group_list.count()):
            item = self.group_list.item(index)
            if item.data(Qt.UserRole) == group_id:
                self.group_list.setCurrentRow(index)
                return

    def refresh_history(self):
        selected_group = self.current_group_id()
        selected_score = self._selected_score_id()
        self._loading_groups = True
        self.group_list.clear()
        groups = score_db.list_groups()
        for group in groups:
            item = QListWidgetItem(group["name"])
            item.setData(Qt.UserRole, group["id"])
            self.group_list.addItem(item)
        if selected_group is not None:
            self._select_group(selected_group)
        if self.group_list.currentRow() < 0 and self.group_list.count():
            self.group_list.setCurrentRow(0)
        self._loading_groups = False
        self._fill_songs(selected_score)
        self._refresh_home_stats()

    def _on_group_changed(self):
        if self._loading_groups:
            return
        self._fill_songs(None)

    def _fill_songs(self, keep_id):
        self.history_list.clear()
        group_id = self.current_group_id()
        if group_id is None:
            return
        query = self.search_songs.text().strip().lower() if hasattr(self, "search_songs") else ""
        for row in score_db.list_scores(group_id):
            if query and query not in (row["title"] or "").lower():
                continue
            prefix = "★  " if row.get("is_favorite") else "   "
            label = f"{prefix}{row['title']}    {row.get('key_sig') or ''}"
            item = QListWidgetItem(label.strip())
            item.setData(Qt.UserRole, row["id"])
            item.setToolTip(
                f"{row['title']}\n{row.get('meter') or '4/4'}  ·  ♩={row.get('bpm') or 100}"
            )
            self.history_list.addItem(item)
            if keep_id is not None and row["id"] == keep_id:
                self.history_list.setCurrentItem(item)

    def create_group(self):
        name, ok = QInputDialog.getText(self, "新建歌单", "歌单名称")
        if not ok:
            return
        try:
            group_id = score_db.create_group(name)
        except ValueError as exc:
            QMessageBox.information(self, "提示", str(exc))
            return
        self.refresh_history()
        self._select_group(group_id)

    def rename_group(self):
        group_id = self.current_group_id()
        if group_id is None:
            return
        name, ok = QInputDialog.getText(
            self,
            "重命名歌单",
            "歌单名称",
            text=(self.group_list.currentItem().text() if self.group_list.currentItem() else ""),
        )
        if not ok:
            return
        try:
            score_db.rename_group(int(group_id), name)
        except ValueError as exc:
            QMessageBox.information(self, "提示", str(exc))
            return
        self.refresh_history()

    def delete_group(self):
        ids = [item.data(Qt.UserRole) for item in self.group_list.selectedItems()]
        if not ids:
            return
        if QMessageBox.question(self, "删除歌单", f"删除所选 {len(ids)} 个歌单？歌曲将移到保留的歌单，不会删除。", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            score_db.delete_groups(ids)
        except ValueError as exc:
            QMessageBox.information(self, "提示", str(exc))
            return
        self.refresh_history()

    def import_playlists(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入歌单", "", "简谱歌单 (*.json)")
        if not path:
            return
        try:
            if Path(path).stat().st_size > 50 * 1024 * 1024:
                raise ValueError("歌单文件不能超过 50 MB。")
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            groups, songs = score_db.import_playlists(payload)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.refresh_history()
        QMessageBox.information(self, "导入完成", f"已导入 {groups} 个歌单、{songs} 首歌曲。同名歌单自动新建副本。")

    def export_selected_playlists(self):
        ids = [item.data(Qt.UserRole) for item in self.group_list.selectedItems()]
        if ids:
            self._export_playlists(ids)

    def export_all_playlists(self):
        self._export_playlists(None)

    def _export_playlists(self, ids):
        path, _ = QFileDialog.getSaveFileName(self, "导出歌单", "简谱歌单.json", "简谱歌单 (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            payload = score_db.export_playlists(ids)
            target = Path(path)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as f:
                temp = Path(f.name)
                json.dump(payload, f, ensure_ascii=False, indent=2)
            try:
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", "歌单、谱面、收藏和歌曲顺序已保存。可用“导入歌单”恢复。")

    def move_history(self):
        score_id = self._selected_score_id()
        if not score_id:
            QMessageBox.information(self, "提示", "请先选一首歌")
            return
        groups = [g for g in score_db.list_groups() if g["id"] != self.current_group_id()]
        if not groups:
            QMessageBox.information(self, "提示", "请先新建另一个歌单")
            return
        name, ok = QInputDialog.getItem(
            self, "移动到歌单", "目标歌单", [g["name"] for g in groups], 0, False
        )
        if not ok:
            return
        target = next(g for g in groups if g["name"] == name)
        score_db.move_score(int(score_id), target["id"])
        self.refresh_history()

    def move_score_to_group(self, score_id, group_id):
        """拖放移动歌曲；移动后仍保持歌曲选中状态。"""
        try:
            score_db.move_score(int(score_id), int(group_id))
        except (TypeError, ValueError):
            return
        self.refresh_history()
        self._select_group(int(group_id))
        self._fill_songs(int(score_id))

    def reorder_history(self):
        """把当前列表的顺序写回数据库，供歌单内拖放使用。"""
        group_id = self.current_group_id()
        if group_id is None:
            return
        ordered = [
            self.history_list.item(index).data(Qt.UserRole)
            for index in range(self.history_list.count())
        ]
        score_db.reorder_scores(int(group_id), ordered)

    def _selected_score_id(self):
        item = self.history_list.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def load_history(self):
        score_id = self._selected_score_id()
        if not score_id:
            QMessageBox.information(self, "提示", "请先在歌单里选一首")
            return None
        row = score_db.get_score(int(score_id))
        if not row:
            return None
        self.stop_sequence()
        self.editing_score_id = int(row["id"])
        self._creating_new_song = False
        self._editing_source_snapshot = row.get("source") or ""
        self.text_input.setPlainText(row["source"])
        self.editing_banner.setText(
            f"正在编辑已有歌曲：{row.get('title') or '未命名'} · 点击转换时会询问保存方式"
        )
        self.set_main_page(2)
        self.lbl_meta.setText(
            f"正在编辑：{row.get('title') or '未命名'}  ·  重新转换会保留这个歌名"
        )
        return row

    def new_song(self):
        self.stop_sequence()
        self.editing_score_id = None
        self._title_override = None
        self._editing_source_snapshot = ""
        self._creating_new_song = True
        self.editing_banner.setText("新歌曲：转换后会保存到当前歌单")
        self.text_input.clear()
        self.key_chart.set_score({}, [])
        self.lbl_meta.setText("新歌曲：粘贴简谱，转换后会保存到当前歌单")
        self.big_hint.setText("准备好创作下一首歌")
        if hasattr(self, "bar_song"):
            self.bar_song.setText("未选择歌曲")

    def play_history(self, mode="preview"):
        row = self.load_history()
        if not row:
            return
        # 播放只解析并载入内存，不算一次保存；否则 update_score 会修改
        # updated_at，歌单按“最近保存”排序时当前歌曲会被自动置顶。
        self.do_convert(show_overwrite_prompt=False, preserve_order=True)
        if self.notes and not self.is_playing:
            self.start_sequence(mode)

    def _step_song(self, delta, force_mode=None):
        count = self.history_list.count()
        if count == 0:
            return
        current = self.history_list.currentRow()
        target = (current + delta) % count
        self.history_list.setCurrentRow(target)
        mode = force_mode or ("global" if self.is_playing and self.play_mode == "global" else "preview")
        self.play_history(mode)

    def previous_song(self, force_global=False):
        self._step_song(-1, "global" if force_global else None)

    def next_song(self, force_global=False):
        self._step_song(1, "global" if force_global else None)

    def delete_history(self):
        ids = [item.data(Qt.UserRole) for item in self.history_list.selectedItems()]
        if not ids:
            return
        if QMessageBox.question(self, "删除歌曲", f"永久删除所选 {len(ids)} 首歌曲？建议先导出备份。", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.stop_sequence()
        score_db.delete_scores(ids)
        if self.editing_score_id in ids:
            self.editing_score_id = None
            self._editing_source_snapshot = ""
            self._creating_new_song = True
            self.editing_banner.setText("原歌曲已删除；再次保存将创建新歌曲")
        self.refresh_history()

    def toggle_favorite(self):
        score_id = self._selected_score_id()
        if not score_id:
            QMessageBox.information(self, "提示", "请先选一首歌")
            return
        row = score_db.get_score(int(score_id))
        if not row:
            return
        score_db.set_favorite(int(score_id), not bool(row.get("is_favorite")))
        self.refresh_history()

    def _find_npm(self):
        node_dirs = [r"C:\app\nodejs", r"C:\Program Files\nodejs"]
        for folder in node_dirs:
            node = os.path.join(folder, "node.exe")
            if os.path.isfile(node):
                return node
        return shutil.which("node")

    def _start_bundled_editor_server(self):
        """Serve the built jpeditor files without requiring Node in installs."""
        if self._jpeditor_server is not None or not JPEDITOR_DIST.is_dir():
            return False

        class EditorHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                # 与开发服务器一致，否则打包后的页面拿不到 SharedArrayBuffer。
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
                super().end_headers()

            def log_message(self, fmt, *args):
                return

        handler = partial(EditorHandler, directory=str(JPEDITOR_DIST))
        try:
            self._jpeditor_server = ThreadingHTTPServer(("127.0.0.1", 1420), handler)
        except OSError:
            self._jpeditor_server = None
            return False
        threading.Thread(
            target=self._jpeditor_server.serve_forever,
            name="jpeditor-static-server",
            daemon=True,
        ).start()
        return True

    def _port_open(self, port: int) -> bool:
        sock = socket.socket()
        sock.settimeout(0.4)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _set_editor_status(self, text):
        self.lbl_editor.setText(text)
        self.lbl_editor.setVisible(bool(text))

    def _open_jpeditor(self):
        if not self._port_open(1420) and JPEDITOR_DIR.is_dir():
            self._editor_url_loaded = False
            if self._start_bundled_editor_server():
                self._set_editor_status("谱面编辑器启动中…")
            else:
                node = self._find_npm()
                vite_js = JPEDITOR_DIR / "node_modules" / "vite" / "bin" / "vite.js"
                if not node or not vite_js.is_file():
                    self._set_editor_status("找不到前端资源或 Node，无法启动谱面编辑器")
                    return
                log_path = score_db.DB_PATH.parent / "jpeditor-dev.log"
                log_file = open(log_path, "w", encoding="utf-8")
                self._jpeditor_proc = subprocess.Popen(
                    [node, str(vite_js), "--host", "127.0.0.1", "--port", "1420", "--strictPort"],
                    cwd=str(JPEDITOR_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                self._set_editor_status("谱面编辑器启动中…")
        if self.editor_view is not None:
            QTimer.singleShot(600, self._load_editor_url)
        else:
            self._set_editor_status("没有内嵌浏览器组件")

    def _load_editor_url(self):
        if self.editor_view is None:
            return
        if self._port_open(1420):
            self._set_editor_status("谱面已嵌入  " + JPEDITOR_URL)
            # Do not reload an already mounted page every time the user opens
            # the editor.  Reloading while the QStackedWidget page is hidden
            # is what makes Qt retain a stale compositor tile from the page
            # that was underneath it.  A restarted Vite server is detected by
            # the explicit flag reset in _open_jpeditor.
            if not self._editor_url_loaded:
                self._editor_url_loaded = True
                self.editor_view.setUrl(QUrl(JPEDITOR_URL))
            else:
                self.editor_view.update()
            return
        if self._server_retries < 40:
            self._server_retries += 1
            self._set_editor_status(f"正在等待编辑器启动… {self._server_retries}")
            QTimer.singleShot(500, self._load_editor_url)
        else:
            self._set_editor_status("编辑器没启动成功，请看 data/jpeditor-dev.log")

    def rename_history(self):
        score_id = self._selected_score_id()
        if not score_id:
            QMessageBox.information(self, "提示", "请先选一首歌")
            return
        row = score_db.get_score(int(score_id))
        if not row:
            return
        name, ok = QInputDialog.getText(self, "修改歌名", "歌名", text=row["title"])
        if not ok:
            return
        try:
            score_db.rename_score(int(score_id), name)
        except ValueError as exc:
            QMessageBox.information(self, "提示", str(exc))
            return
        self.refresh_history()

    def open_score_page(self):
        text = self.text_input.toPlainText().strip()
        if text:
            self.send_score_to_editor(text)
        try:
            ready = self._ensure_editor_view()
        except Exception as exc:
            self._set_editor_status("谱面编辑器初始化失败：" + str(exc))
            QMessageBox.warning(self, "打开谱面失败", "内嵌浏览器初始化失败，请重新启动程序后再试。")
            return
        if not ready:
            QMessageBox.warning(self, "打开谱面失败", "内嵌浏览器组件没有加载成功，请重新安装后再试。")
            return
        self._page_before_editor = self.main_pages.currentIndex()
        # Expose the web view before starting/loading it.  QWebEngine paints
        # reliably when its first navigation happens in a visible widget;
        # loading it while hidden in QStackedWidget leaves mixed old/new tiles
        # on some Qt WebEngine builds.
        self.stack.setCurrentWidget(self.editor_page)
        self.editor_view.setUpdatesEnabled(True)
        self._open_jpeditor()

    def close_score_page(self):
        if self.editor_view is not None:
            self.editor_view.setUpdatesEnabled(False)
        self.stack.setCurrentIndex(0)
        self.set_main_page(self._page_before_editor)

    def send_current_to_editor(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先粘贴简谱")
            return
        self.send_score_to_editor(text)

    def send_score_to_editor(self, text: str):
        self._pending_editor_text = text
        self._editor_retries = 0
        if self.editor_view is None:
            return
        self._push_editor_text()

    def _on_editor_loaded(self, ok: bool):
        if ok:
            # A newly exposed QWebEngineView may otherwise keep the backing
            # store marked dirty until the next event loop turn, which makes
            # the first frame look like a blend of the previous page.
            self.editor_view.setUpdatesEnabled(True)
            self.editor_view.update()
        if ok and self._pending_editor_text:
            self._fit_editor_to_window()
            self._editor_retries = 0
            self._push_editor_text()

    def _push_editor_text(self):
        if self.editor_view is None or not self._pending_editor_text:
            return
        payload = json.dumps(self._pending_editor_text, ensure_ascii=False)
        js = (
            "(function(){"
            f"if (window.__loadJianpu) {{ window.__loadJianpu({payload}); return true; }}"
            "return false;"
            "})()"
        )
        self.editor_view.page().runJavaScript(js, self._after_editor_push)

    def _after_editor_push(self, ok):
        if ok or self._editor_retries >= 40:
            return
        self._editor_retries += 1
        QTimer.singleShot(500, self._push_editor_text)

    def do_convert(self, show_overwrite_prompt=True, preserve_order=False):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先粘贴简谱")
            return

        # 最后一层保护：用户可能直接从歌单选中歌曲后进入创作页，没有点击“载入”。
        # 只要没有明确点“新歌曲”，转换前就把当前选中的歌曲视为编辑目标。
        if not self.editing_score_id and not self._creating_new_song:
            selected_id = self._selected_score_id()
            if selected_id:
                selected = score_db.get_score(int(selected_id))
                if selected and text:
                    self.editing_score_id = int(selected["id"])
                    self._editing_source_snapshot = selected.get("source") or ""
                    self.editing_banner.setText(
                        f"正在编辑已有歌曲：{selected.get('title') or '未命名'} · 点击转换时会询问保存方式"
                    )

        if self.editing_score_id and show_overwrite_prompt:
            current = score_db.get_score(int(self.editing_score_id))
            if current:
                dialog = QMessageBox(self)
                dialog.setWindowTitle("这首歌已经存在")
                dialog.setText(f"你正在编辑「{current.get('title') or '未命名'}」。")
                dialog.setInformativeText("请选择保存方式，避免不小心覆盖原曲。")
                overwrite = dialog.addButton("覆盖原曲", QMessageBox.AcceptRole)
                duplicate = dialog.addButton("另存为新曲", QMessageBox.ActionRole)
                cancel = dialog.addButton("取消", QMessageBox.RejectRole)
                dialog.setWindowModality(Qt.ApplicationModal)
                dialog.exec_()
                clicked = dialog.clickedButton()
                if clicked == cancel:
                    return
                if clicked == duplicate:
                    default_name = f"{current.get('title') or '未命名'}（副本）"
                    title, ok = QInputDialog.getText(self, "另存为新曲", "新歌名", text=default_name)
                    if not ok or not title.strip():
                        return
                    self._title_override = title.strip()
                    self.editing_score_id = None
                    self.editing_banner.setText(f"已选择另存为新曲：{title.strip()}")
                else:
                    self._title_override = None

        self.meta, self.notes = convert_jianpu(text)
        valid = len([n for n in self.notes if not n.get("is_rest") and n.get("key")])

        self.key_chart.set_score(self.meta, self.notes)
        self.set_main_page(3)
        if hasattr(self, "bar_song"):
            self.bar_song.setText(self._title_override or self.meta.get("title") or "未命名")

        tie_count = self.meta.get("tie_count", 0)
        bad_bars = self.meta.get("bad_bars") or []
        bar_note = (
            f"有 {len(bad_bars)} 小节时值对不上"
            if bad_bars
            else f"{self.meta.get('bar_count', 0)} 小节时值已核对"
        )
        self.lbl_meta.setText(
            f"标题: {self.meta.get('title')}　|　调号: {self.meta.get('key')}　|　"
            f"拍号: {self.meta.get('meter')}　|　速度: ♩ = {self.meta.get('bpm')}　|　"
            f"延音合并 {tie_count} 处　|　{bar_note}"
        )
        self.lbl_progress.setText(f"已转换 {valid} 个音符")
        self.big_hint.setText(
            f"括号里的同音是延音，只按一次。已合并 {tie_count} 处"
        )
        if self.editing_score_id:
            # 重新转换已载入歌曲时只更新谱面，保留用户改过的歌名和歌单。
            score_db.update_score(
                self.editing_score_id,
                text,
                self.meta.get("key") or "",
                self.meta.get("meter") or "",
                int(self.meta.get("bpm") or 0),
                valid,
                reorder=not preserve_order,
            )
        else:
            self.editing_score_id = score_db.save_score(
                self._title_override or self.meta.get("title") or "未命名",
                text,
                self.meta.get("key") or "",
                self.meta.get("meter") or "",
                int(self.meta.get("bpm") or 0),
                valid,
                self.current_group_id(),
            )
            self._title_override = None
            self._creating_new_song = False
        self._editing_source_snapshot = text
        if self.editing_score_id:
            current = score_db.get_score(int(self.editing_score_id))
            if current:
                self.editing_banner.setText(
                    f"正在编辑已有歌曲：{current.get('title') or '未命名'} · 已保存"
                )
        self.refresh_history()
        self.send_score_to_editor(text)
        self.big_hint.setStyleSheet(self._hint_style("idle"))

    def start_sequence(self, mode):
        if not self.notes:
            QMessageBox.information(self, "提示", "请先转换简谱")
            return
        if self.is_playing and self.play_mode == mode:
            self.stop_sequence()
            return
        if self.is_playing:
            self.stop_sequence()
        self.countdown_timer.stop()
        self.play_mode = mode
        self.is_playing = True
        self.play_index = 0
        self.btn_play.setText("暂停预览" if mode == "preview" else "预览播放")
        self.btn_global.setText("停止全局" if mode == "global" else "全局播放")
        if hasattr(self, "bar_mode"):
            self.bar_mode.setText("全局演奏中" if mode == "global" else "预览播放中")
        self._play_next_note()

    def stop_sequence(self):
        self.is_playing = False
        self.play_timer.stop()
        self.countdown_timer.stop()
        self.global_keys.release()
        self.btn_play.setText("预览播放")
        self.btn_global.setText("全局播放")
        if hasattr(self, "bar_mode"):
            self.bar_mode.setText("预览模式")
        self.lbl_progress.setText("已停止")
        self._clear_key_highlight()
        self.key_chart.set_active(-1)
        self.big_hint.setText("播放已停止")
        self.big_hint.setStyleSheet(self._hint_style("idle"))

    def _play_next_note(self):
        if not self.is_playing or self.play_index >= len(self.notes):
            self.stop_sequence()
            self.lbl_progress.setText("播放完毕")
            return

        note = self.notes[self.play_index]
        current_index = self.play_index
        self.play_index += 1
        self.key_chart.set_active(current_index)

        compensate = self.speed_slider.value() / 100.0
        duration_ms = int(note.get("duration_ms", 300) / compensate)
        duration_ms = max(40, duration_ms)

        if note.get("is_rest") or not note.get("key"):
            self.lbl_mod.value_label.setText("休止")
            self.lbl_note.value_label.setText("0")
            self.lbl_freq.value_label.setText("---")
            self.lbl_key.value_label.setText("---")
            self.big_hint.setText(f"休止  ({duration_ms}ms)")
            self.big_hint.setStyleSheet(self._hint_style("rest"))
            self._clear_key_highlight()
            self.global_keys.release()
        else:
            self._play_one_note(note, duration_ms)

        self.lbl_progress.setText(f"播放中 {self.play_index}/{len(self.notes)}")
        self.play_timer.start(duration_ms + 12)

    def _play_one_note(self, note, duration_ms):
        key = note["key"]
        mouse = note.get("mouse")
        num = note.get("num", "1")
        freq_base = note.get("freq_base", num)

        base = BASE_FREQ.get(freq_base, BASE_FREQ.get(num, 440))
        octave = int(note.get("octave", 0))
        accidental = int(note.get("accidental", 0))
        freq = int(round(base * (2 ** octave) * (2 ** (accidental / 12.0))))

        mod_name = note.get("mod_label") or {
            "left": "降调", "middle": "半调", "right": "升调"
        }.get(mouse, "正常")
        self.lbl_mod.value_label.setText(mod_name)
        self.lbl_note.value_label.setText(note["desc"])
        self.lbl_freq.value_label.setText(f"{freq} Hz")
        self.lbl_key.value_label.setText(key)

        tie_txt = "  延音" if note.get("tied") else ""
        grace_txt = "  倚音" if note.get("is_grace") else ""
        self.big_hint.setText(f"{key} → {note['desc']}{tie_txt}{grace_txt}   {freq}Hz")
        self.big_hint.setStyleSheet(self._hint_style("live"))

        self._highlight_key(key)
        if self.play_mode == "global":
            self.global_keys.hold(key, mouse)
        else:
            self.player.play(freq, duration_ms)

    def _highlight_key(self, key):
        self._clear_key_highlight()
        if key and key in self.key_buttons:
            self.key_buttons[key].setStyleSheet(self._key_style(True))
            self.active_key = key

    def _clear_key_highlight(self):
        if self.active_key and self.active_key in self.key_buttons:
            self.key_buttons[self.active_key].setStyleSheet(self._key_style(False))
        self.active_key = None

    # ---------- 实时键盘演奏 ----------
    def get_current_mod(self):
        buttons = QApplication.mouseButtons()
        if buttons & Qt.LeftButton: return "left"
        if buttons & Qt.MidButton:  return "middle"
        if buttons & Qt.RightButton: return "right"
        return "none"

    def get_freq(self, base_note, mod):
        base = BASE_FREQ.get(base_note, 440)
        if mod == "left":
            return 262 if base_note == "1·" else int(base / 2)
        if mod == "middle":
            return int(round(base * (2 ** (1/12))))
        if mod == "right":
            return int(base * 2)
        return base

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.text().upper()
        if event.key() == Qt.Key_Comma:
            key = ","
        if key not in KEY_MAP:
            return

        base_note = KEY_MAP[key]
        mod = self.get_current_mod()
        freq = self.get_freq(base_note, mod)

        mod_names = {"none": "正常", "left": "降调", "middle": "半调", "right": "升调"}
        display = base_note
        if mod == "left":
            display = "1" if base_note == "1·" else f"·{base_note}"
        elif mod == "middle":
            display = f"#{base_note}"
        elif mod == "right":
            display = base_note + "·"

        self.lbl_mod.value_label.setText(mod_names[mod])
        self.lbl_note.value_label.setText(display)
        self.lbl_freq.value_label.setText(f"{freq} Hz")
        self.lbl_key.value_label.setText(key)

        self.big_hint.setText(f"{key} → {display}   {freq}Hz")
        self.big_hint.setStyleSheet(self._hint_style("live"))
        self._highlight_key(key)
        self.player.play(freq, 260)

    def keyReleaseEvent(self, event):
        key = event.text().upper()
        if event.key() == Qt.Key_Comma:
            key = ","
        if key == self.active_key:
            self._clear_key_highlight()
            self.big_hint.setStyleSheet(self._hint_style("idle"))

    def closeEvent(self, event):
        self.countdown_timer.stop()
        if self._jpeditor_server is not None:
            self._jpeditor_server.shutdown()
            self._jpeditor_server.server_close()
            self._jpeditor_server = None
        if self._jpeditor_proc is not None and self._jpeditor_proc.poll() is None:
            self._jpeditor_proc.terminate()
        try:
            hwnd = int(self.winId())
            for hotkey_id in (1, 2, 3, 4):
                _user32.UnregisterHotKey(hwnd, hotkey_id)
        except Exception:
            pass
        self.global_keys.release()
        super().closeEvent(event)


if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DeltaHarmonica.Jianpu")
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    ensure_pick_server()
    app = QApplication(sys.argv)
    _load_webengine()
    if _utf8_codec is not None:
        QTextCodec.setCodecForLocale(_utf8_codec)
    app.setWindowIcon(MainWindow._load_app_icon(None))
    win = MainWindow()
    win.show()
    win.setFocus()
    sys.exit(app.exec_())
