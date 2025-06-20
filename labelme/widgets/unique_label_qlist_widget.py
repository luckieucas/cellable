# -*- encoding: utf-8 -*-

import html
from typing import Optional, Tuple


from qtpy import QtWidgets
from qtpy.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets


from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    # ----------- 查找 -----------
    def findItemByLabel(self, label: str):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item
        return None

    # ----------- 生成彩色圆点图标的小工具 -----------
    @staticmethod
    def _color_icon(rgb: tuple[int, int, int]) -> QtGui.QIcon:
        """Return a 12×12 circular icon filled with rgb."""
        pix = QtGui.QPixmap(12, 12)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtGui.QColor(*rgb))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(0, 0, 11, 11)
        p.end()
        return QtGui.QIcon(pix)

    # ----------- 创建条目 -----------
    def createItemFromLabel(
            self,
            label: str,
            rgb: Optional[Tuple[int, int, int]] = None,
            checked: bool = True
    ) -> QtWidgets.QListWidgetItem:
        """
        新建一个带复选框的条目；若 rgb 给出，则显示彩色圆点。
        """
        if self.findItemByLabel(label):
            raise ValueError(f"Item for label '{label}' already exists")

        item = QtWidgets.QListWidgetItem(label)
        item.setData(Qt.UserRole, label)

        # 让它可勾选
        flags = (item.flags() | Qt.ItemIsUserCheckable |
                 Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setFlags(flags)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

        # 彩色圆点
        if rgb is not None:
            item.setData(Qt.DecorationRole, self._color_icon(rgb))

        return item

    # ----------- 更新显示（改为直接改 text / icon） -----------
    def setItemLabel(self, item: QtWidgets.QListWidgetItem,
                     label: str, color: Optional[Tuple[int, int, int]] = None):
        item.setText(label)
        if color is not None:
            item.setData(Qt.DecorationRole, self._color_icon(color))
