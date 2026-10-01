import sys
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ==================== 音高 ====================
BASE_FREQ = {
    '1': 262,
    '2': 294,
    '3': 330,
    '4': 349,
    '5': 393,
    '6': 440,
    '7': 494,
    '1·': 524,
}

KEY_MAP = {
    'Z': '1',
    'X': '2',
    'C': '3',
    'V': '4',
    'B': '5',
    'N': '6',
    'M': '7',
    '.': '1·',
}

KEY_ORDER = ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '.']


class BeepPlayer:
    def play(self, freq, duration_ms=260):
        def _beep():
            try:
                import winsound
                winsound.Beep(int(freq), max(int(duration_ms), 80))
            except Exception as e:
                print("Beep error:", e)
        threading.Thread(target=_beep, daemon=True).start()


class HarmonicaWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.player = BeepPlayer()
        self.active_key = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('三角洲口风琴模拟器')
        self.setMinimumSize(900, 780)
        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f1a;
                color: #e0e0e0;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # ========== 标题 ==========
        title = QLabel('harmonica  三角洲口风琴')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #ff6b81;
            letter-spacing: 2px;
            padding: 6px;
        """)
        main_layout.addWidget(title)

        # ========== 状态栏 ==========
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border-radius: 12px;
                border: 1px solid #2a2a4a;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 12, 20, 12)

        self.lbl_mod = self._status_label('调式', '正常')
        self.lbl_note = self._status_label('音符', '---')
        self.lbl_freq = self._status_label('频率', '--- Hz')
        self.lbl_key = self._status_label('按键', '---')

        status_layout.addWidget(self.lbl_mod)
        status_layout.addWidget(self.lbl_note)
        status_layout.addWidget(self.lbl_freq)
        status_layout.addWidget(self.lbl_key)
        main_layout.addWidget(status_frame)

        # ========== 当前演奏提示 ==========
        self.big_hint = QLabel('按下键盘开始演奏')
        self.big_hint.setAlignment(Qt.AlignCenter)
        self.big_hint.setFixedHeight(70)
        self.big_hint.setStyleSheet("""
            background: #1a1a2e;
            border-radius: 14px;
            border: 2px solid #2a2a4a;
            font-size: 26px;
            font-weight: bold;
            color: #7ec8e3;
        """)
        main_layout.addWidget(self.big_hint)

        # ========== 可视化键盘 ==========
        key_frame = QFrame()
        key_frame.setStyleSheet("""
            QFrame {
                background: #16162b;
                border-radius: 16px;
                border: 1px solid #2a2a4a;
            }
        """)
        key_layout = QVBoxLayout(key_frame)
        key_layout.setContentsMargins(20, 16, 20, 16)
        key_layout.setSpacing(12)

        key_title = QLabel('键盘映射（按住鼠标左/中/右键可切换调式）')
        key_title.setStyleSheet("color: #888; font-size: 13px;")
        key_layout.addWidget(key_title)

        self.key_buttons = {}
        row = QHBoxLayout()
        row.setSpacing(10)

        for k in KEY_ORDER:
            btn = QPushButton(f'{k}\n{KEY_MAP[k]}')
            btn.setFixedSize(86, 86)
            btn.setEnabled(False)  # 仅作显示
            btn.setStyleSheet(self._key_style(False))
            self.key_buttons[k] = btn
            row.addWidget(btn)

        key_layout.addLayout(row)
        main_layout.addWidget(key_frame)

        # ========== 操作说明 ==========
        help_frame = QFrame()
        help_frame.setStyleSheet("""
            QFrame {
                background: #16162b;
                border-radius: 14px;
                border: 1px solid #2a2a4a;
            }
        """)
        help_layout = QVBoxLayout(help_frame)
        help_layout.setContentsMargins(18, 14, 18, 14)

        help_title = QLabel('操作方式')
        help_title.setStyleSheet("color: #ff6b81; font-size: 15px; font-weight: bold;")
        help_layout.addWidget(help_title)

        help_text = QLabel(
            '① 先按住鼠标左键（降调）/ 中键（半调）/ 右键（升调）不放\n'
            '② 再按下键盘 Z X C V B N M 或 . 发声\n'
            '③ 不按鼠标时为正常音调'
        )
        help_text.setStyleSheet("color: #bbb; font-size: 14px; line-height: 1.6;")
        help_layout.addWidget(help_text)
        main_layout.addWidget(help_frame)

        # ========== 底部提示 ==========
        footer = QLabel('声音目前使用系统蜂鸣（winsound），真实口风琴音色需要采样文件支持')
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #555; font-size: 12px; padding-top: 4px;")
        main_layout.addWidget(footer)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _status_label(self, title, value):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("color: #888; font-size: 12px;")
        t.setAlignment(Qt.AlignCenter)
        v = QLabel(value)
        v.setObjectName('value')
        v.setStyleSheet("color: #fff; font-size: 16px; font-weight: bold;")
        v.setAlignment(Qt.AlignCenter)
        l.addWidget(t)
        l.addWidget(v)
        w.value_label = v
        return w

    def _key_style(self, active=False):
        if active:
            return """
                QPushButton {
                    background: #e94560;
                    color: white;
                    border: 2px solid #ff6b81;
                    border-radius: 12px;
                    font-size: 15px;
                    font-weight: bold;
                }
            """
        return """
            QPushButton {
                background: #2a2a4a;
                color: #ddd;
                border: 2px solid #3a3a5a;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
        """

    def get_freq(self, base_note, mod):
        base = BASE_FREQ.get(base_note, 440)
        if mod == 'left':
            return 262 if base_note == '1·' else int(base / 2)
        if mod == 'mid':
            return int(round(base * (2 ** (1 / 12))))
        if mod == 'right':
            return int(base * 2)
        return base

    def get_current_mod(self):
        buttons = QApplication.mouseButtons()
        if buttons & Qt.LeftButton:
            return 'left'
        if buttons & Qt.MidButton:
            return 'mid'
        if buttons & Qt.RightButton:
            return 'right'
        return 'none'

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        key = event.text().upper()
        if key not in KEY_MAP:
            if event.key() == Qt.Key_Period:
                key = '.'
            else:
                return

        base_note = KEY_MAP[key]
        mod = self.get_current_mod()
        freq = self.get_freq(base_note, mod)

        mod_names = {
            'none': '正常',
            'left': '降调（低八度）',
            'mid': '半调（升半音）',
            'right': '升调（高八度）'
        }

        # 显示用的音符名
        if mod == 'left':
            display = '1' if base_note == '1·' else f'·{base_note}'
        elif mod == 'mid':
            display = f'#{base_note}'
        elif mod == 'right':
            display = base_note + '·'
        else:
            display = base_note

        # 更新状态
        self.lbl_mod.value_label.setText(mod_names[mod])
        self.lbl_note.value_label.setText(display)
        self.lbl_freq.value_label.setText(f'{freq} Hz')
        self.lbl_key.value_label.setText(key)

        self.big_hint.setText(f'{key}  →  {display}    {freq} Hz')
        self.big_hint.setStyleSheet("""
            background: #1a1a2e;
            border-radius: 14px;
            border: 2px solid #e94560;
            font-size: 26px;
            font-weight: bold;
            color: #ff6b81;
        """)

        # 高亮按键
        if self.active_key and self.active_key in self.key_buttons:
            self.key_buttons[self.active_key].setStyleSheet(self._key_style(False))
        self.active_key = key
        if key in self.key_buttons:
            self.key_buttons[key].setStyleSheet(self._key_style(True))

        self.player.play(freq, 280)

    def keyReleaseEvent(self, event):
        key = event.text().upper()
        if event.key() == Qt.Key_Period:
            key = '.'
        if key == self.active_key and key in self.key_buttons:
            self.key_buttons[key].setStyleSheet(self._key_style(False))
            self.active_key = None
            self.big_hint.setStyleSheet("""
                background: #1a1a2e;
                border-radius: 14px;
                border: 2px solid #2a2a4a;
                font-size: 26px;
                font-weight: bold;
                color: #7ec8e3;
            """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = HarmonicaWidget()
    w.show()
    w.setFocus()
    sys.exit(app.exec_())