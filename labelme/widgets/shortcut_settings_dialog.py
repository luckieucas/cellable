"""Keyboard shortcuts settings widget and dialog."""

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.config import get_default_config
from labelme.config import get_user_config_path
from labelme.config import save_shortcuts as config_save_shortcuts
from labelme.config.shortcut_schema import SHORTCUT_SCHEMA
from labelme.logger import logger


# Public API
__all__ = ["ShortcutSettingsWidget", "ShortcutSettingsDialog"]


def _to_key_sequence(value):
    """Convert config value (str, list, or None) to QKeySequence for display."""
    if value is None:
        return QtGui.QKeySequence()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not value:
        return QtGui.QKeySequence()
    return QtGui.QKeySequence(value)


def _from_key_sequence(seq):
    """Convert QKeySequence to config string, or None if empty."""
    s = seq.toString().strip()
    return s if s else None


class _ShortcutEditEventFilter(QtCore.QObject):
    """Event filter to clear shortcut on double-click."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick:
            if hasattr(obj, "setKeySequence"):
                obj.setKeySequence(QtGui.QKeySequence())
            return True
        return super().eventFilter(obj, event)


class ShortcutSettingsWidget(QtWidgets.QWidget):
    """Widget for customizing keyboard shortcuts. Used in dock or dialog."""

    shortcutsSaved = QtCore.Signal()

    def __init__(self, parent=None, shortcuts=None, config_path=None, on_save=None):
        super().__init__(parent)
        self._shortcuts = dict(shortcuts) if shortcuts else {}
        self._config_path = config_path or get_user_config_path()
        self._default_shortcuts = get_default_config().get("shortcuts", {})
        self._edits = {}
        self._on_save = on_save

        layout = QtWidgets.QVBoxLayout(self)

        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Search shortcuts...")
        self._search_edit.textChanged.connect(self._filter_table)
        search_layout.addWidget(self._search_edit)
        layout.addLayout(search_layout)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        self._conflict_label = QtWidgets.QLabel()
        self._conflict_label.setStyleSheet("color: #c00;")
        self._conflict_label.setWordWrap(True)
        layout.addWidget(self._conflict_label)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self._clear_btn = QtWidgets.QPushButton("Clear Shortcut")
        self._clear_btn.setToolTip("Clear the shortcut for the selected action")
        self._clear_btn.clicked.connect(self._clear_selected_shortcut)
        self._clear_btn.setEnabled(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        btn_layout.addWidget(self._clear_btn)
        reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        hint_label = QtWidgets.QLabel(
            "Double-click shortcut cell or use Clear Shortcut to remove a shortcut."
        )
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint_label)

        self._populate_table()
        self._check_conflicts()

    def _populate_table(self):
        self._table.setRowCount(len(SHORTCUT_SCHEMA))
        for row, (config_key, label) in enumerate(SHORTCUT_SCHEMA):
            item = QtWidgets.QTableWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, config_key)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 0, item)

            value = self._shortcuts.get(config_key, self._default_shortcuts.get(config_key))
            edit = QtWidgets.QKeySequenceEdit(_to_key_sequence(value))
            if hasattr(edit, "setClearButtonEnabled"):
                edit.setClearButtonEnabled(True)
            edit.installEventFilter(_ShortcutEditEventFilter(edit))
            edit.keySequenceChanged.connect(self._on_shortcut_changed)
            self._table.setCellWidget(row, 1, edit)
            self._edits[config_key] = edit

    def _on_selection_changed(self):
        """Enable/disable Clear Shortcut button based on row selection."""
        has_selection = len(self._table.selectedItems()) > 0
        self._clear_btn.setEnabled(has_selection)

    def _clear_selected_shortcut(self):
        """Clear the shortcut for the currently selected row."""
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        config_key = item.data(QtCore.Qt.UserRole)
        if config_key and config_key in self._edits:
            self._edits[config_key].setKeySequence(QtGui.QKeySequence())
            self._check_conflicts()

    def _filter_table(self):
        filter_text = self._search_edit.text().strip().lower()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item:
                label = item.text().lower()
                show = not filter_text or filter_text in label
                self._table.setRowHidden(row, not show)

    def _on_shortcut_changed(self):
        self._check_conflicts()

    def _check_conflicts(self):
        key_to_actions = {}
        for config_key, edit in self._edits.items():
            seq = edit.keySequence()
            if not seq.isEmpty():
                s = _from_key_sequence(seq)
                if s:
                    key_to_actions.setdefault(s, []).append(config_key)
        conflicts = [(k, v) for k, v in key_to_actions.items() if len(v) > 1]
        if conflicts:
            msg = "Conflict: the following shortcuts are assigned to multiple actions: "
            msg += "; ".join("{} -> {}".format(k, ", ".join(v)) for k, v in conflicts)
            self._conflict_label.setText(msg)
        else:
            self._conflict_label.setText("")

    def _reset_to_defaults(self):
        for config_key, edit in self._edits.items():
            value = self._default_shortcuts.get(config_key)
            edit.setKeySequence(_to_key_sequence(value))
        self._check_conflicts()

    def _save(self):
        shortcuts = self.get_shortcuts()
        # Save ALL current form values so they persist (fixes reverts).
        # Include every schema key: non-None = user value, None = cleared (revert to default).
        to_save = {}
        for config_key in self._edits:
            value = shortcuts.get(config_key)
            to_save[config_key] = value
        try:
            config_save_shortcuts(self._config_path, to_save)
            self.shortcutsSaved.emit()
            if self._on_save:
                self._on_save()
            QtWidgets.QMessageBox.information(
                self, "Saved", "Keyboard shortcuts saved. Changes applied.",
            )
        except Exception as e:
            logger.error("Failed to save shortcuts: %s", e)
            QtWidgets.QMessageBox.critical(
                self, "Save Failed", "Failed to save shortcuts: {}".format(e),
            )

    def get_shortcuts(self):
        """Return current shortcut values from the form (all schema keys)."""
        result = {}
        for config_key, edit in self._edits.items():
            value = _from_key_sequence(edit.keySequence())
            result[config_key] = value
        return result

    def set_shortcuts(self, shortcuts):
        """Update displayed shortcuts (e.g. after config reload)."""
        self._shortcuts = dict(shortcuts) if shortcuts else {}
        for config_key, edit in self._edits.items():
            value = self._shortcuts.get(config_key, self._default_shortcuts.get(config_key))
            edit.setKeySequence(_to_key_sequence(value))
        self._check_conflicts()


class ShortcutSettingsDialog(QtWidgets.QDialog):
    """Modal dialog wrapper for ShortcutSettingsWidget (legacy)."""

    def __init__(self, parent=None, shortcuts=None, config_path=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(550, 500)
        self.resize(600, 550)
        self._widget = ShortcutSettingsWidget(
            self, shortcuts=shortcuts, config_path=config_path,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._widget)
        cancel_btn = QtWidgets.QPushButton("Close")
        cancel_btn.clicked.connect(self.accept)
        layout.addWidget(cancel_btn, alignment=QtCore.Qt.AlignRight)
        self._widget.shortcutsSaved.connect(self.accept)

    def get_shortcuts(self):
        return self._widget.get_shortcuts()
