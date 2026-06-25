"""The control center window and ``control-center`` entry point.

A native Qt app: pick a command on the left, fill its auto-generated form on the right, Run it (as a
child process) while its output streams into the log pane, and save/load the form as a named preset.
Every command in :data:`osc_genai.cli_spec.REGISTRY` appears automatically — there is no
per-command UI code here.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from osc_genai.cli_spec import REGISTRY
from osc_genai.gui import presets
from osc_genai.gui.forms import CommandForm
from osc_genai.gui.invoke import build_invocation
from osc_genai.gui.workers import CommandProcess

_KIND_ORDER = ["realtime", "training", "oneshot"]
_KIND_LABEL = {"realtime": "Realtime", "training": "Training", "oneshot": "Tools"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("osc-genai control center")
        self.resize(1000, 720)
        self.process = CommandProcess(self)
        self.process.output.connect(self._append_log)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)

        self._forms: dict[str, CommandForm] = {}
        self._index_by_command: dict[str, int] = {}

        self._build_ui()
        self._select_first()

    # -- construction -------------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.command_list = QListWidget()
        self.command_list.setMaximumWidth(260)
        self.command_list.currentItemChanged.connect(self._on_command_changed)

        self.stack = QStackedWidget()
        for name in self._ordered_commands():
            form = CommandForm(REGISTRY[name])
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form)
            self._index_by_command[name] = self.stack.addWidget(scroll)
            self._forms[name] = form
        self._populate_command_list()

        self.title = QLabel()
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        self.title.setFont(title_font)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: gray;")

        # preset row
        self.preset_combo = QComboBox()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_preset)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_preset)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_preset)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(load_btn)
        preset_row.addWidget(save_btn)
        preset_row.addWidget(delete_btn)

        # run row
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.process.stop)
        self.stop_btn.setEnabled(False)
        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch(1)
        run_row.addWidget(clear_btn)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.title)
        right_layout.addWidget(self.description)
        right_layout.addWidget(self.stack, 1)
        right_layout.addLayout(preset_row)
        right_layout.addLayout(run_row)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.command_list)
        top.addWidget(right)
        top.setStretchFactor(1, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo"))
        self.log.setMaximumBlockCount(5000)

        outer = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(top)
        outer.addWidget(self.log)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 2)
        self.setCentralWidget(outer)

    def _ordered_commands(self) -> list[str]:
        return [
            name
            for kind in _KIND_ORDER
            for name in sorted(REGISTRY)
            if REGISTRY[name].kind == kind
        ]

    def _populate_command_list(self) -> None:
        for kind in _KIND_ORDER:
            header = QListWidgetItem(_KIND_LABEL[kind].upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setForeground(Qt.GlobalColor.gray)
            self.command_list.addItem(header)
            for name in sorted(REGISTRY):
                if REGISTRY[name].kind != kind:
                    continue
                item = QListWidgetItem("   " + name)
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.command_list.addItem(item)

    def _select_first(self) -> None:
        for row in range(self.command_list.count()):
            item = self.command_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole):
                self.command_list.setCurrentRow(row)
                return

    # -- selection / presets ------------------------------------------------------------------

    @property
    def current_command(self) -> str | None:
        item = self.command_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_command_changed(self, current: QListWidgetItem | None, _prev=None) -> None:
        name = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not name:
            return
        cmd = REGISTRY[name]
        self.stack.setCurrentIndex(self._index_by_command[name])
        self.title.setText(cmd.name)
        self.description.setText(cmd.description)
        self._refresh_presets()

    def _refresh_presets(self) -> None:
        self.preset_combo.clear()
        if self.current_command:
            self.preset_combo.addItems(presets.list_presets(self.current_command))

    def _load_preset(self) -> None:
        name = self.current_command
        preset = self.preset_combo.currentText()
        if not name or not preset:
            return
        self._forms[name].set_values(presets.load_preset(name, preset))

    def _save_preset(self) -> None:
        name = self.current_command
        if not name:
            return
        preset, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not preset.strip():
            return
        try:
            values = self._forms[name].values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        presets.save_preset(name, preset.strip(), values)
        self._refresh_presets()
        self.preset_combo.setCurrentText(preset.strip())

    def _delete_preset(self) -> None:
        name = self.current_command
        preset = self.preset_combo.currentText()
        if name and preset:
            presets.delete_preset(name, preset)
            self._refresh_presets()

    # -- running ------------------------------------------------------------------------------

    def _run(self) -> None:
        name = self.current_command
        if not name or self.process.running:
            return
        form = self._forms[name]
        missing = form.missing_required()
        if missing:
            QMessageBox.warning(
                self, "Missing required", "Fill required flag(s): " + ", ".join(missing)
            )
            return
        try:
            values = form.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        program, args = build_invocation(name, values)
        self._append_log(
            f"$ {program} {' '.join(args[2:])}\n"
        )  # skip the -c <code> preamble
        self.process.start(name, values)

    def _on_started(self) -> None:
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_finished(self, code: int) -> None:
        self._append_log(f"\n[exited with code {code}]\n")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.process.running:
            self.process.stop()
        event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
