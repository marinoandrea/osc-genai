"""Render a :class:`~osc_genai.cli_spec.Command` as a Qt form, and read/populate its values.

The form is generated entirely from the registry: there is no per-command UI code, so a new flag in
``cli_spec`` appears here automatically. Widgets are chosen from each :class:`Param`'s ``kind`` and
``widget`` hint (checkbox / dropdown / line edit / path picker).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from osc_genai.cli_spec import Command, Param

_SAVE_PARAMS = {
    "out",
    "save",
}  # path params that name an output (save dialog, not open)


class _PathField(QWidget):
    """A line edit plus a browse button that opens the right file/dir dialog for a path param."""

    def __init__(self, param: Param) -> None:
        super().__init__()
        self._param = param
        self.edit = QLineEdit()
        if param.default:
            self.edit.setText(str(param.default))
        button = QPushButton("…")
        button.setFixedWidth(32)
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(button)

    def _browse(self) -> None:
        if self._param.widget == "dir":
            path = QFileDialog.getExistingDirectory(self, f"Select {self._param.name}")
        elif self._param.name in _SAVE_PARAMS:
            path, _ = QFileDialog.getSaveFileName(self, f"Select {self._param.name}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self._param.name}")
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, value: str) -> None:
        self.edit.setText(value)


class CommandForm(QWidget):
    """A generated form for one command. ``values()`` reads it; ``set_values()`` populates it."""

    def __init__(self, command: Command) -> None:
        super().__init__()
        self.command = command
        self._widgets: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        layout.setLabelAlignment(layout.labelAlignment())
        for p in command.params:
            widget = self._make_widget(p)
            self._widgets[p.name] = widget
            label = p.flag + (" *" if p.required else "")
            if p.help:
                widget.setToolTip(p.help)
            layout.addRow(label, widget)

    def _make_widget(self, p: Param) -> QWidget:
        if p.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(p.effective_default))
            return box
        if p.ui_choices:
            combo = QComboBox()
            combo.addItems(list(p.ui_choices))
            if p.default in p.ui_choices:
                combo.setCurrentText(str(p.default))
            return combo
        if p.widget in {"file", "dir", "checkpoint"}:
            return _PathField(p)
        edit = QLineEdit()
        if p.default is not None and p.nargs != "+":
            edit.setText(str(p.default))
        edit.setPlaceholderText("" if p.default is None else str(p.default))
        return edit

    def _coerce(self, p: Param, text: str) -> Any:
        text = text.strip()
        if not text:
            return p.effective_default
        if p.nargs == "+":
            return text.split()
        if p.kind == "int":
            return int(text)
        if p.kind == "float":
            return float(text)
        return text

    def values(self) -> dict[str, Any]:
        """Current form values as a name->value dict (coerced to each param's type).

        Raises ``ValueError`` if a numeric field has unparseable text.
        """
        out: dict[str, Any] = {}
        for p in self.command.params:
            widget = self._widgets[p.name]
            if isinstance(widget, QCheckBox):
                out[p.name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                out[p.name] = widget.currentText()
            else:  # QLineEdit or _PathField
                try:
                    out[p.name] = self._coerce(p, widget.text())
                except ValueError as exc:
                    raise ValueError(f"{p.flag}: {exc}") from exc
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        """Populate the form from a values dict (e.g. a loaded preset). Unknown keys are ignored."""
        for p in self.command.params:
            if p.name not in values:
                continue
            value = values[p.name]
            widget = self._widgets[p.name]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif value is None:
                widget.setText("")
            elif p.nargs == "+" and isinstance(value, list):
                widget.setText(" ".join(str(x) for x in value))
            else:
                widget.setText(str(value))

    def missing_required(self) -> list[str]:
        """Names of required params left empty (so the UI can block a run with a clear message)."""
        missing = []
        values = self.values()
        for p in self.command.params:
            if p.required and not values.get(p.name):
                missing.append(p.flag)
        return missing
