# -*- encoding: utf-8 -*-

import html
from typing import Optional, Tuple, Dict, List, Set
import numpy as np

from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal
from PyQt5 import QtCore, QtGui, QtWidgets

from .escapable_qlist_widget import EscapableQListWidget
from labelme.label_state import LabelState, LabelMetadataStore


# State badge colors
STATE_COLORS = {
    LabelState.PROPOSED: (150, 150, 150),   # Gray
    LabelState.EDITED: (255, 165, 0),       # Orange
    LabelState.VERIFIED: (50, 205, 50),     # Green
}


class UniqueLabelQListWidget(EscapableQListWidget):
    # 新增信号，用于通知label可见性改变
    labelVisibilityChanged = Signal(str, bool)  # label, visible
    
    # Signals for label lifecycle actions
    labelVerifyRequested = Signal(str)      # label_id
    labelUnverifyRequested = Signal(str)    # label_id
    labelRejectRequested = Signal(str)      # label_id
    labelRevertRequested = Signal(str)      # label_id
    
    # Signals for visibility quick actions
    soloCurrentRequested = Signal(str)      # label_id
    showAllRequested = Signal()
    
    # Signal for navigation
    labelDoubleClicked = Signal(str)        # label_id - emitted on double-click to jump to label

    def __init__(self):
        super().__init__()
        self.tiff_mask = None  # 存储3D mask数据用于计算voxel size
        self.label_voxel_counts = {}  # 存储每个label的voxel count
        self._item_by_label: Dict[str, QtWidgets.QListWidgetItem] = {}
        self._metadata_store: Optional[LabelMetadataStore] = None
        self._visibility_manager = None  # Will be set by app.py
        
        # Track which labels are hidden due to state (for display purposes)
        self._state_hidden_labels: Set[str] = set()
        
        # Track all labels (including filtered out ones)
        self._all_labels_data: Dict[str, Dict] = {}  # label -> {rgb, checked, ...}
        
        # 连接item changed信号
        self.itemChanged.connect(self._on_item_changed)
        
        # 连接双击信号
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def addItem(self, item):
        super().addItem(item)
        if isinstance(item, QtWidgets.QListWidgetItem):
            label = item.data(Qt.UserRole)
            if label is not None:
                self._item_by_label[str(label)] = item

    def takeItem(self, row):
        item = super().takeItem(row)
        if item is not None:
            label = item.data(Qt.UserRole)
            if label is not None:
                key = str(label)
                if self._item_by_label.get(key) is item:
                    self._item_by_label.pop(key, None)
        return item

    def clear(self):
        self._item_by_label.clear()
        super().clear()

    def _on_item_changed(self, item):
        """处理item的checkbox状态改变"""
        if item is None:
            return
        label = item.data(Qt.UserRole)
        visible = (item.checkState() == Qt.Checked)
        
        # Update visibility manager if available
        if self._visibility_manager:
            self._visibility_manager.set_user_visible(label, visible, emit_signal=False)
        
        self.labelVisibilityChanged.emit(label, visible)
    
    def _on_item_double_clicked(self, item):
        """Handle double-click on item to jump to label's middle slice."""
        if item is None:
            return
        label = item.data(Qt.UserRole)
        if label:
            self.labelDoubleClicked.emit(label)
    
    def set_metadata_store(self, store: LabelMetadataStore):
        """Set the metadata store for label state tracking."""
        self._metadata_store = store
        self._update_all_state_badges()
    
    def set_visibility_manager(self, manager):
        """Set the visibility manager for state-based visibility tracking."""
        self._visibility_manager = manager
        # Connect to visibility manager signals
        if manager:
            manager.viewHiddenStatesChanged.connect(self._on_view_hidden_states_changed)
            manager.allVisibilityChanged.connect(self._update_all_hidden_indicators)
            manager.effectiveVisibilityChanged.connect(self._on_effective_visibility_changed)
    
    def _on_view_hidden_states_changed(self, hidden_states: set):
        """Handle changes to which states are hidden in views."""
        self._update_state_hidden_labels()
        self._update_all_hidden_indicators()
    
    def _on_effective_visibility_changed(self, label: str, visible: bool):
        """Handle effective visibility change for a single label."""
        self._update_label_hidden_indicator(label)
    
    def _update_state_hidden_labels(self):
        """Update the set of labels hidden due to their state."""
        self._state_hidden_labels.clear()
        if not self._visibility_manager or not self._metadata_store:
            return
        
        for row in range(self.count()):
            item = self.item(row)
            if item:
                label = item.data(Qt.UserRole)
                if self._visibility_manager.is_hidden_by_state(label):
                    self._state_hidden_labels.add(label)
    
    def _update_all_hidden_indicators(self):
        """Update hidden indicators for all items."""
        self._update_state_hidden_labels()
        for row in range(self.count()):
            item = self.item(row)
            if item:
                label = item.data(Qt.UserRole)
                self._update_item_hidden_indicator(item, label)
    
    def _update_label_hidden_indicator(self, label: str):
        """Update hidden indicator for a specific label."""
        item = self.findItemByLabel(label)
        if item:
            self._update_item_hidden_indicator(item, label)
    
    def _update_item_hidden_indicator(self, item: QtWidgets.QListWidgetItem, label: str):
        """Update the hidden indicator for an item."""
        if not self._visibility_manager:
            return
        
        is_hidden_by_state = self._visibility_manager.is_hidden_by_state(label)
        
        # Update tooltip to indicate hidden status
        if is_hidden_by_state:
            item.setToolTip(f"Label {label} is hidden in views (state-based)")
            # Add visual indicator - strikethrough or faded text
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            # Add eye-off icon or indicator in text
            self._update_item_text_with_hidden(item, label, hidden=True)
        else:
            item.setToolTip("")
            font = item.font()
            font.setItalic(False)
            item.setFont(font)
            self._update_item_text_with_hidden(item, label, hidden=False)
    
    def _update_item_text_with_hidden(self, item: QtWidgets.QListWidgetItem, label: str, hidden: bool):
        """Update item text to show hidden indicator."""
        voxel_count = self.label_voxel_counts.get(label, 0)
        state = None
        if self._metadata_store:
            state = self._metadata_store.get_state(label)
        state_indicator = self._get_state_indicator(state) if state else ""
        
        # Hidden indicator
        hidden_indicator = "👁‍🗨 " if hidden else ""  # Eye with speech bubble = hidden
        
        if state_indicator:
            display_text = f"{hidden_indicator}{state_indicator} {label} ({voxel_count} voxels)" if voxel_count > 0 else f"{hidden_indicator}{state_indicator} {label}"
        else:
            display_text = f"{hidden_indicator}{label} ({voxel_count} voxels)" if voxel_count > 0 else f"{hidden_indicator}{label}"
        
        # Block signals to avoid triggering callbacks
        self.blockSignals(True)
        item.setText(display_text)
        self.blockSignals(False)
    
    def _show_context_menu(self, position):
        """Show context menu with label lifecycle actions."""
        item = self.itemAt(position)
        if item is None:
            return
        
        label = item.data(Qt.UserRole)
        if label is None:
            return
        
        menu = QtWidgets.QMenu(self)
        
        # Get current state from metadata store
        current_state = None
        can_revert = False
        if self._metadata_store:
            current_state = self._metadata_store.get_state(label)
            can_revert = self._metadata_store.can_revert(label)
        
        # === Visibility Actions ===
        visibility_menu = menu.addMenu("👁 Visibility")
        
        # Solo current
        solo_action = visibility_menu.addAction("Solo Current (S)")
        solo_action.setToolTip("Show only this label in views")
        solo_action.triggered.connect(lambda: self.soloCurrentRequested.emit(label))
        
        # Show all
        show_all_action = visibility_menu.addAction("Show All")
        show_all_action.setToolTip("Show all labels in views")
        show_all_action.triggered.connect(lambda: self.showAllRequested.emit())
        
        # Show hidden status
        if self._visibility_manager and self._visibility_manager.is_hidden_by_state(label):
            visibility_menu.addSeparator()
            hidden_info = visibility_menu.addAction("⚠ Hidden due to VERIFIED state")
            hidden_info.setEnabled(False)
        
        menu.addSeparator()
        
        # === Lifecycle Actions ===
        # Verify action
        verify_action = menu.addAction("✓ Verify (V)")
        verify_action.setEnabled(current_state != LabelState.VERIFIED)
        verify_action.triggered.connect(lambda: self.labelVerifyRequested.emit(label))
        
        # Unverify action
        unverify_action = menu.addAction("↩ Unverify")
        unverify_action.setEnabled(current_state == LabelState.VERIFIED)
        unverify_action.triggered.connect(lambda: self.labelUnverifyRequested.emit(label))
        
        menu.addSeparator()
        
        # Revert to Proposed action
        revert_action = menu.addAction("⟲ Revert to Proposed (R)")
        revert_action.setEnabled(can_revert and current_state != LabelState.PROPOSED)
        revert_action.triggered.connect(lambda: self.labelRevertRequested.emit(label))
        
        menu.addSeparator()
        
        # Reject (delete) action
        reject_action = menu.addAction("✗ Reject / Delete (Del)")
        reject_action.triggered.connect(lambda: self.labelRejectRequested.emit(label))
        
        menu.addSeparator()
        
        # === Navigation ===
        goto_action = menu.addAction("🎯 Go to Middle Slice")
        goto_action.setToolTip("Jump to the middle slice of this label (double-click)")
        goto_action.triggered.connect(lambda: self.labelDoubleClicked.emit(label))
        
        # Show state info
        if current_state:
            menu.addSeparator()
            state_info = menu.addAction(f"State: {current_state.value.upper()}")
            state_info.setEnabled(False)
        
        menu.exec_(self.mapToGlobal(position))

    def set_tiff_mask(self, mask):
        """设置3D mask数据，用于计算voxel size"""
        self.tiff_mask = mask
        self._calculate_voxel_counts()
        self._update_display()

    def set_label_voxel_counts(self, counts: Dict[str, int]):
        """Set precomputed voxel counts without scanning the full mask again."""
        self.label_voxel_counts = {str(label): int(count) for label, count in counts.items()}
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
        """更新显示，添加voxel count和state信息"""
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                voxel_count = self.label_voxel_counts.get(label, 0)
                
                # Get state indicator
                state = None
                if self._metadata_store:
                    state = self._metadata_store.get_state(label)
                state_indicator = self._get_state_indicator(state) if state else ""
                
                # 更新显示文本，包含voxel count和state
                if state_indicator:
                    display_text = f"{state_indicator} {label} ({voxel_count} voxels)"
                else:
                    display_text = f"{label} ({voxel_count} voxels)"
                item.setText(display_text)
                
                # Update background color
                if state:
                    bg_color = self._get_state_background_color(state)
                    if bg_color:
                        item.setBackground(QtGui.QBrush(bg_color))
                    else:
                        item.setBackground(QtGui.QBrush())

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

    def sort_by_label_id(self, ascending=True):
        """按label ID排序 (1, 2, 3, ...)"""
        # 收集所有items及其信息
        items_data = []
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                rgb = self._extract_color_from_item(item)
                checked = (item.checkState() == Qt.Checked)
                # 将label转换为整数用于排序
                try:
                    label_id = int(label)
                except ValueError:
                    label_id = 0  # 非数字label排在最前面
                items_data.append((label, label_id, rgb, checked))
        
        # 按label ID排序
        items_data.sort(key=lambda x: x[1], reverse=not ascending)
        
        # 清空列表并重新添加排序后的items
        self.clear()
        for label, label_id, rgb, checked in items_data:
            # 重新创建item
            item = self.createItemFromLabel(label, rgb=rgb, checked=checked)
            self.addItem(item)
    
    def export_label_voxel_tsv(self, include_hidden=True) -> str:
        """Export label ID and voxel size as TSV for spreadsheet paste."""
        lines = ["label_id\tvoxel_size"]
        for row in range(self.count()):
            item = self.item(row)
            if item is None:
                continue
            if not include_hidden and item.isHidden():
                continue
            label = str(item.data(Qt.UserRole))
            voxel_count = int(self.label_voxel_counts.get(label, 0))
            lines.append(f"{label}\t{voxel_count}")
        return "\n".join(lines)
    
    def sort_by_state(self, order: List[LabelState] = None):
        """Sort labels by their state (e.g., PROPOSED first, then EDITED, then VERIFIED)."""
        if order is None:
            order = [LabelState.PROPOSED, LabelState.EDITED, LabelState.VERIFIED]
        
        def state_priority(label):
            if self._metadata_store:
                state = self._metadata_store.get_state(label)
                if state in order:
                    return order.index(state)
            return len(order)
        
        # 收集所有items及其信息
        items_data = []
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                rgb = self._extract_color_from_item(item)
                checked = (item.checkState() == Qt.Checked)
                priority = state_priority(label)
                items_data.append((label, priority, rgb, checked))
        
        # 按state priority排序
        items_data.sort(key=lambda x: x[1])
        
        # 清空列表并重新添加排序后的items
        self.clear()
        for label, priority, rgb, checked in items_data:
            item = self.createItemFromLabel(label, rgb=rgb, checked=checked)
            self.addItem(item)
    
    # ---- List Filtering by State ----
    
    def apply_state_filter(self, filter_mode):
        """
        Apply state-based filtering to show/hide items in the list.
        This affects which items are VISIBLE in the list UI, not the underlying data.
        
        Args:
            filter_mode: LabelFilterMode enum value
        """
        from labelme.label_visibility import LabelFilterMode
        
        for row in range(self.count()):
            item = self.item(row)
            if item:
                label = item.data(Qt.UserRole)
                passes_filter = True
                
                if filter_mode != LabelFilterMode.ALL:
                    state = None
                    if self._metadata_store:
                        state = self._metadata_store.get_state(label)
                    passes_filter = filter_mode.matches_state(state)
                
                item.setHidden(not passes_filter)
    
    def get_visible_item_count(self) -> int:
        """Get count of items not hidden by filtering."""
        count = 0
        for row in range(self.count()):
            item = self.item(row)
            if item and not item.isHidden():
                count += 1
        return count
    
    def sync_checkbox_with_visibility_manager(self):
        """Sync all checkbox states with the visibility manager."""
        if not self._visibility_manager:
            return
        
        self.blockSignals(True)
        for row in range(self.count()):
            item = self.item(row)
            if item:
                label = item.data(Qt.UserRole)
                user_visible = self._visibility_manager.get_user_visible(label)
                item.setCheckState(Qt.Checked if user_visible else Qt.Unchecked)
        self.blockSignals(False)
        self._update_all_hidden_indicators()

    def _extract_color_from_item(self, item):
        """从item中提取颜色信息"""
        icon = item.data(Qt.DecorationRole)
        if icon is not None:
            # 这里简化处理，实际可能需要从icon中提取颜色
            # 暂时返回None，让调用方使用默认颜色
            return None
        return None

    def _update_all_state_badges(self):
        """Update state badges for all items based on metadata store."""
        for row in range(self.count()):
            item = self.item(row)
            if item is not None:
                label = item.data(Qt.UserRole)
                self._update_item_state_badge(item, label)
    
    def _update_item_state_badge(self, item: QtWidgets.QListWidgetItem, label: str):
        """Update the state badge for a single item."""
        if self._metadata_store is None:
            return
        
        state = self._metadata_store.get_state(label)
        if state is None:
            return
        
        # Update the item's background or text to show state
        # We'll use a compound icon approach: color circle + state indicator
        color_icon = item.data(Qt.DecorationRole)
        if color_icon:
            # Get the original color from the existing icon if possible
            pass
        
        # Update display text with state indicator
        voxel_count = self.label_voxel_counts.get(label, 0)
        state_indicator = self._get_state_indicator(state)
        display_text = f"{state_indicator} {label} ({voxel_count} voxels)" if voxel_count > 0 else f"{state_indicator} {label}"
        
        # Block signals to avoid triggering _on_item_changed
        self.blockSignals(True)
        item.setText(display_text)
        
        # Set background color based on state (subtle tint)
        bg_color = self._get_state_background_color(state)
        if bg_color:
            item.setBackground(QtGui.QBrush(bg_color))
        else:
            item.setBackground(QtGui.QBrush())
        
        self.blockSignals(False)
    
    @staticmethod
    def _get_state_indicator(state: LabelState) -> str:
        """Get a text indicator for the label state."""
        if state == LabelState.PROPOSED:
            return "○"  # Empty circle - proposed
        elif state == LabelState.EDITED:
            return "◐"  # Half-filled circle - edited
        elif state == LabelState.VERIFIED:
            return "●"  # Filled circle - verified
        return ""
    
    @staticmethod
    def _get_state_background_color(state: LabelState) -> Optional[QtGui.QColor]:
        """Get a subtle background color for the state."""
        if state == LabelState.PROPOSED:
            return QtGui.QColor(200, 200, 200, 40)   # Light gray tint
        elif state == LabelState.EDITED:
            return QtGui.QColor(255, 200, 100, 40)  # Light orange tint
        elif state == LabelState.VERIFIED:
            return QtGui.QColor(100, 220, 100, 40)  # Light green tint
        return None
    
    def update_label_state(self, label: str):
        """Update the display for a single label's state."""
        item = self.findItemByLabel(label)
        if item:
            self._update_item_state_badge(item, label)

    # ----------- 查找 -----------
    def findItemByLabel(self, label: str):
        label = str(label)
        cached = self._item_by_label.get(label)
        if cached is not None:
            return cached
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                self._item_by_label[label] = item
                return item
        return None

    def remove_label_fast(self, label: str) -> bool:
        """Remove one label row from the list and drop its cached voxel count."""
        label = str(label)
        item = self.findItemByLabel(label)
        if item is None:
            return False
        row = self.row(item)
        if row < 0:
            return False
        self.takeItem(row)
        self.label_voxel_counts.pop(label, None)
        self._all_labels_data.pop(label, None)
        return True

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
    
    @staticmethod
    def _color_icon_with_state(rgb: tuple[int, int, int], state: LabelState) -> QtGui.QIcon:
        """Return a 16×12 icon with color circle and state indicator."""
        pix = QtGui.QPixmap(24, 12)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw color circle
        p.setBrush(QtGui.QColor(*rgb))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(0, 0, 11, 11)
        
        # Draw state indicator
        state_color = STATE_COLORS.get(state, (150, 150, 150))
        p.setBrush(QtGui.QColor(*state_color))
        if state == LabelState.PROPOSED:
            # Empty circle
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(*state_color), 1.5))
            p.drawEllipse(14, 2, 7, 7)
        elif state == LabelState.EDITED:
            # Half-filled circle
            p.setPen(QtCore.Qt.NoPen)
            p.drawPie(14, 2, 7, 7, 0, 180 * 16)
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(*state_color), 1.5))
            p.drawEllipse(14, 2, 7, 7)
        else:  # VERIFIED
            # Filled circle
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(14, 2, 7, 7)
        
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
        Also shows state badge if metadata store is available.
        """
        label = str(label)
        existing_item = self.findItemByLabel(label)
        if existing_item:
            # 如果item已存在，更新其属性并返回
            state = None
            if self._metadata_store:
                state = self._metadata_store.get_state(label)
            
            if rgb is not None:
                if state:
                    existing_item.setData(Qt.DecorationRole, self._color_icon_with_state(rgb, state))
                else:
                    existing_item.setData(Qt.DecorationRole, self._color_icon(rgb))
            existing_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            
            # 更新显示文本 with state indicator
            voxel_count = self.label_voxel_counts.get(label, 0)
            state_indicator = self._get_state_indicator(state) if state else ""
            if state_indicator:
                display_text = f"{state_indicator} {label} ({voxel_count} voxels)" if voxel_count > 0 else f"{state_indicator} {label}"
            else:
                display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
            existing_item.setText(display_text)
            
            # Update background color
            if state:
                bg_color = self._get_state_background_color(state)
                if bg_color:
                    existing_item.setBackground(QtGui.QBrush(bg_color))
                else:
                    existing_item.setBackground(QtGui.QBrush())
            
            return existing_item

        # Get state from metadata store
        state = None
        if self._metadata_store:
            state = self._metadata_store.get_state(label)

        # 获取voxel count信息
        voxel_count = self.label_voxel_counts.get(label, 0)
        state_indicator = self._get_state_indicator(state) if state else ""
        if state_indicator:
            display_text = f"{state_indicator} {label} ({voxel_count} voxels)" if voxel_count > 0 else f"{state_indicator} {label}"
        else:
            display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
        
        item = QtWidgets.QListWidgetItem(display_text)
        item.setData(Qt.UserRole, label)

        # 让它可勾选
        flags = (item.flags() | Qt.ItemIsUserCheckable |
                 Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setFlags(flags)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

        # 彩色圆点 with state indicator
        if rgb is not None:
            if state:
                item.setData(Qt.DecorationRole, self._color_icon_with_state(rgb, state))
            else:
                item.setData(Qt.DecorationRole, self._color_icon(rgb))
        
        # Set background color based on state
        if state:
            bg_color = self._get_state_background_color(state)
            if bg_color:
                item.setBackground(QtGui.QBrush(bg_color))
        
        # Register with visibility manager
        if self._visibility_manager:
            self._visibility_manager.register_label(label)
            # Sync checkbox state from visibility manager
            user_visible = self._visibility_manager.get_user_visible(label)
            item.setCheckState(Qt.Checked if user_visible else Qt.Unchecked)

        return item

    # ----------- 更新显示（改为直接改 text / icon） -----------
    def setItemLabel(self, item: QtWidgets.QListWidgetItem,
                     label: str, color: Optional[Tuple[int, int, int]] = None):
        label = str(label)
        # 获取voxel count信息
        voxel_count = self.label_voxel_counts.get(label, 0)
        
        # Get state indicator
        state = None
        if self._metadata_store:
            state = self._metadata_store.get_state(label)
        state_indicator = self._get_state_indicator(state) if state else ""
        
        if state_indicator:
            display_text = f"{state_indicator} {label} ({voxel_count} voxels)" if voxel_count > 0 else f"{state_indicator} {label}"
        else:
            display_text = f"{label} ({voxel_count} voxels)" if voxel_count > 0 else label
        item.setText(display_text)
        
        if color is not None:
            if state:
                item.setData(Qt.DecorationRole, self._color_icon_with_state(color, state))
            else:
                item.setData(Qt.DecorationRole, self._color_icon(color))
        
        # Update background color
        if state:
            bg_color = self._get_state_background_color(state)
            if bg_color:
                item.setBackground(QtGui.QBrush(bg_color))
            else:
                item.setBackground(QtGui.QBrush())
