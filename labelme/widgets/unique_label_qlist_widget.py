# -*- encoding: utf-8 -*-

import html
from typing import Optional, Tuple
import numpy as np

from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal
from PyQt5 import QtCore, QtGui, QtWidgets

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    # 新增信号，用于通知label可见性改变
    labelVisibilityChanged = Signal(str, bool)  # label, visible

    def __init__(self):
        super().__init__()
        self.tiff_mask = None  # 存储3D mask数据用于计算voxel size
        self.label_voxel_counts = {}  # 存储每个label的voxel count
        
        # 连接item changed信号
        self.itemChanged.connect(self._on_item_changed)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def _on_item_changed(self, item):
        """处理item的checkbox状态改变"""
        if item is None:
            return
        label = item.data(Qt.UserRole)
        visible = (item.checkState() == Qt.Checked)
        self.labelVisibilityChanged.emit(label, visible)

    def set_tiff_mask(self, mask):
        """设置3D mask数据，用于计算voxel size"""
        self.tiff_mask = mask
        self._calculate_voxel_counts()
        self._update_display()

    def _calculate_voxel_counts(self):
        """计算每个label的voxel数量"""
        if self.tiff_mask is None:
            return
        
        unique_labels, counts = np.unique(self.tiff_mask, return_counts=True)
        self.label_voxel_counts = {}
        
        for label, count in zip(unique_labels, counts):
            if label > 0:  # 跳过背景(label=0)
                self.label_voxel_counts[str(label)] = count

    def _update_display(self):
        """更新显示，添加voxel count信息"""
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                voxel_count = self.label_voxel_counts.get(label, 0)
                # 更新显示文本，包含voxel count
                display_text = f"{label} ({voxel_count} voxels)"
                item.setText(display_text)

    def sort_by_voxel_size(self, ascending=False):
        """按voxel size排序"""
        if not self.label_voxel_counts:
            return
        
        # 收集所有items及其信息
        items_data = []
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                voxel_count = self.label_voxel_counts.get(label, 0)
                rgb = self._extract_color_from_item(item)
                checked = (item.checkState() == Qt.Checked)
                items_data.append((label, voxel_count, rgb, checked))
        
        # 按voxel count排序
        items_data.sort(key=lambda x: x[1], reverse=not ascending)
        
        # 清空列表并重新添加排序后的items
        self.clear()
        for label, voxel_count, rgb, checked in items_data:
            # 重新创建item
            item = self.createItemFromLabel(label, rgb=rgb, checked=checked)
            self.addItem(item)

    def _extract_color_from_item(self, item):
        """从item中提取颜色信息"""
        icon = item.data(Qt.DecorationRole)
        if icon is not None:
            # 这里简化处理，实际可能需要从icon中提取颜色
            # 暂时返回None，让调用方使用默认颜色
            return None
        return None

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
        如果label已存在，则返回已存在的item。
        """
        existing_item = self.findItemByLabel(label)
        if existing_item:
            # 如果item已存在，更新其属性并返回
            if rgb is not None:
                existing_item.setData(Qt.DecorationRole, self._color_icon(rgb))
            existing_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            # 更新显示文本
            voxel_count = self.label_voxel_counts.get(label, 0)
            display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
            existing_item.setText(display_text)
            return existing_item

        # 获取voxel count信息
        voxel_count = self.label_voxel_counts.get(label, 0)
        display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
        
        item = QtWidgets.QListWidgetItem(display_text)
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
        # 获取voxel count信息
        voxel_count = self.label_voxel_counts.get(label, 0)
        display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
        item.setText(display_text)
        
        if color is not None:
            item.setData(Qt.DecorationRole, self._color_icon(color))
