# labelme/label_visibility.py
# -*- coding: utf-8 -*-
"""
Label visibility management for state-based filtering and view hiding.

This module provides:
- LabelFilterMode enum for list filtering options
- LabelVisibilityManager for centralized visibility logic
"""

import json
from enum import Enum
from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field

from qtpy.QtCore import QObject, Signal

from labelme.label_state import LabelState, LabelMetadataStore


class LabelFilterMode(Enum):
    """Filter modes for which labels appear in the label list."""
    ALL = "all"
    PROPOSED = "proposed"
    EDITED = "edited"
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"  # Proposed + Edited
    
    def __str__(self):
        return self.value
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            LabelFilterMode.ALL: "All",
            LabelFilterMode.PROPOSED: "Proposed",
            LabelFilterMode.EDITED: "Edited",
            LabelFilterMode.VERIFIED: "Verified",
            LabelFilterMode.NOT_VERIFIED: "Not Verified",
        }
        return names.get(self, self.value)
    
    def matches_state(self, state: Optional[LabelState]) -> bool:
        """Check if a label state matches this filter mode.
        
        Labels without a state (None) are treated as PROPOSED/unverified.
        """
        if self == LabelFilterMode.ALL:
            return True
        elif self == LabelFilterMode.PROPOSED:
            # None (no metadata) is treated as PROPOSED
            return state == LabelState.PROPOSED or state is None
        elif self == LabelFilterMode.EDITED:
            return state == LabelState.EDITED
        elif self == LabelFilterMode.VERIFIED:
            return state == LabelState.VERIFIED
        elif self == LabelFilterMode.NOT_VERIFIED:
            # None (no metadata) is treated as not verified
            return state in (LabelState.PROPOSED, LabelState.EDITED, None)
        return True


class LabelVisibilityManager(QObject):
    """
    Manages label visibility with two independent concepts:
    
    1. list_filter_mode: Which labels appear in the Label List panel
    2. view_hidden_states: Which label states are hidden from 2D/3D views
    
    Effective visibility = user_checkbox_visible AND (state NOT in view_hidden_states)
    
    Signals:
    - listFilterChanged: Emitted when the list filter mode changes
    - viewHiddenStatesChanged: Emitted when view hidden states change
    - effectiveVisibilityChanged: Emitted when any label's effective visibility changes
    - soloModeChanged: Emitted when solo mode is toggled
    """
    
    # Signals
    listFilterChanged = Signal(object)  # LabelFilterMode
    viewHiddenStatesChanged = Signal(set)  # Set of LabelState to hide
    effectiveVisibilityChanged = Signal(str, bool)  # label_id, effective_visible
    allVisibilityChanged = Signal()  # Batch signal when multiple labels change
    soloModeChanged = Signal(bool, str)  # is_solo, solo_label_id (empty if not solo)
    
    def __init__(self, metadata_store: Optional[LabelMetadataStore] = None):
        super().__init__()
        
        self._metadata_store = metadata_store
        
        # List filter: which labels appear in the list
        self._list_filter_mode = LabelFilterMode.ALL
        
        # View hidden states: which states are hidden from views
        self._view_hidden_states: Set[LabelState] = {LabelState.VERIFIED}  # Default: hide verified
        
        # Per-label user checkbox state (True = user wants it visible)
        self._user_visible: Dict[str, bool] = {}
        
        # Solo mode
        self._solo_mode = False
        self._solo_label: str = ""
        
        # Cache of all known labels
        self._all_labels: Set[str] = set()
    
    def set_metadata_store(self, store: LabelMetadataStore):
        """Set the metadata store for state lookups."""
        self._metadata_store = store
    
    # ---- List Filter ----
    
    @property
    def list_filter_mode(self) -> LabelFilterMode:
        """Get the current list filter mode."""
        return self._list_filter_mode
    
    def get_list_filter_mode(self) -> LabelFilterMode:
        """Get the current list filter mode (method version)."""
        return self._list_filter_mode
    
    def set_list_filter_mode(self, mode: LabelFilterMode):
        """Set the list filter mode."""
        if self._list_filter_mode != mode:
            self._list_filter_mode = mode
            self.listFilterChanged.emit(mode)
    
    def label_passes_list_filter(self, label_id: str) -> bool:
        """Check if a label passes the current list filter."""
        if self._list_filter_mode == LabelFilterMode.ALL:
            return True
        
        state = self._get_label_state(label_id)
        return self._list_filter_mode.matches_state(state)
    
    def get_filtered_labels(self, all_labels: List[str]) -> List[str]:
        """Filter a list of labels based on current filter mode."""
        return [l for l in all_labels if self.label_passes_list_filter(l)]
    
    # ---- View Hidden States ----
    
    @property
    def view_hidden_states(self) -> Set[LabelState]:
        """Get the set of states that are hidden in views."""
        return self._view_hidden_states.copy()
    
    def get_view_hidden_states(self) -> Set[LabelState]:
        """Get the set of states that are hidden in views (method version)."""
        return self._view_hidden_states.copy()
    
    @property
    def hide_verified_in_views(self) -> bool:
        """Check if VERIFIED labels are hidden in views."""
        return LabelState.VERIFIED in self._view_hidden_states
    
    def set_hide_verified_in_views(self, hide: bool):
        """Set whether VERIFIED labels are hidden in views."""
        if hide:
            if LabelState.VERIFIED not in self._view_hidden_states:
                self._view_hidden_states.add(LabelState.VERIFIED)
                self._emit_visibility_changes_for_state(LabelState.VERIFIED)
                self.viewHiddenStatesChanged.emit(self._view_hidden_states)
        else:
            if LabelState.VERIFIED in self._view_hidden_states:
                self._view_hidden_states.discard(LabelState.VERIFIED)
                self._emit_visibility_changes_for_state(LabelState.VERIFIED)
                self.viewHiddenStatesChanged.emit(self._view_hidden_states)
    
    def set_view_hidden_states(self, states: Set[LabelState]):
        """Set the states that should be hidden in views."""
        if self._view_hidden_states != states:
            old_states = self._view_hidden_states
            self._view_hidden_states = states.copy()
            
            # Emit changes for affected states
            changed_states = old_states.symmetric_difference(states)
            for state in changed_states:
                self._emit_visibility_changes_for_state(state)
            
            self.viewHiddenStatesChanged.emit(self._view_hidden_states)
    
    def is_state_hidden_in_views(self, state: Optional[LabelState]) -> bool:
        """Check if a state is hidden in views."""
        return state in self._view_hidden_states
    
    # ---- User Checkbox Visibility ----
    
    def get_user_visible(self, label_id: str) -> bool:
        """Get the user's checkbox visibility setting for a label."""
        return self._user_visible.get(str(label_id), True)  # Default to visible
    
    def set_user_visible(self, label_id: str, visible: bool, emit_signal: bool = True):
        """Set the user's checkbox visibility setting for a label."""
        label_id = str(label_id)
        old_visible = self._user_visible.get(label_id, True)
        self._user_visible[label_id] = visible
        self._all_labels.add(label_id)
        
        if emit_signal and old_visible != visible:
            effective = self.get_effective_visible(label_id)
            self.effectiveVisibilityChanged.emit(label_id, effective)
    
    def set_all_user_visible(self, visible: bool):
        """Set visibility for all known labels."""
        for label_id in self._all_labels:
            self._user_visible[label_id] = visible
        
        # Exit solo mode if showing all
        if visible and self._solo_mode:
            self._solo_mode = False
            self._solo_label = ""
            self.soloModeChanged.emit(False, "")
        
        self.allVisibilityChanged.emit()
    
    def register_label(self, label_id: str):
        """Register a label as known (for tracking)."""
        label_id = str(label_id)
        self._all_labels.add(label_id)
        if label_id not in self._user_visible:
            self._user_visible[label_id] = True
    
    def unregister_label(self, label_id: str):
        """Unregister a label."""
        label_id = str(label_id)
        self._all_labels.discard(label_id)
        self._user_visible.pop(label_id, None)
    
    # ---- Effective Visibility (combines all factors) ----
    
    def get_effective_visible(self, label_id: str) -> bool:
        """
        Get the effective visibility for a label.
        
        effective_visible = user_checkbox_visible 
                           AND (label_state NOT in view_hidden_states)
                           AND (NOT solo_mode OR label == solo_label)
        """
        label_id = str(label_id)
        
        # Solo mode check
        if self._solo_mode:
            if label_id != self._solo_label:
                return False
        
        # User checkbox check
        if not self.get_user_visible(label_id):
            return False
        
        # State-based hiding check
        state = self._get_label_state(label_id)
        if state in self._view_hidden_states:
            return False
        
        return True
    
    def is_hidden_by_state(self, label_id: str) -> bool:
        """Check if a label is hidden due to its state (not user checkbox)."""
        state = self._get_label_state(label_id)
        return state in self._view_hidden_states
    
    def get_hidden_label_ids(self) -> Set[str]:
        """Get all label IDs that are currently hidden in views."""
        hidden = set()
        for label_id in self._all_labels:
            if not self.get_effective_visible(label_id):
                hidden.add(label_id)
        return hidden
    
    def get_visible_label_ids(self) -> Set[str]:
        """Get all label IDs that are currently visible in views."""
        visible = set()
        for label_id in self._all_labels:
            if self.get_effective_visible(label_id):
                visible.add(label_id)
        return visible
    
    # ---- Solo Mode ----
    
    def is_solo_mode(self) -> bool:
        """Check if solo mode is active."""
        return self._solo_mode
    
    @property
    def solo_mode_active(self) -> bool:
        """Property version: Check if solo mode is active."""
        return self._solo_mode
    
    @property
    def solo_label(self) -> str:
        """Get the label being shown in solo mode."""
        return self._solo_label
    
    def set_solo_mode(self, label_id: str):
        """Enter solo mode showing only the specified label."""
        label_id = str(label_id)
        self._solo_mode = True
        self._solo_label = label_id
        self.soloModeChanged.emit(True, label_id)
        self.allVisibilityChanged.emit()
    
    def exit_solo_mode(self):
        """Exit solo mode."""
        if self._solo_mode:
            self._solo_mode = False
            self._solo_label = ""
            self.soloModeChanged.emit(False, "")
            self.allVisibilityChanged.emit()
    
    def clear_solo_mode(self):
        """Clear/exit solo mode (alias for exit_solo_mode)."""
        self.exit_solo_mode()
    
    def toggle_solo_mode(self, label_id: str):
        """Toggle solo mode for a label."""
        if self._solo_mode and self._solo_label == label_id:
            self.exit_solo_mode()
        else:
            self.set_solo_mode(label_id)
    
    # ---- Quick Actions ----
    
    def show_all(self):
        """Show all labels (reset user visibility and exit solo mode)."""
        self._solo_mode = False
        self._solo_label = ""
        for label_id in self._all_labels:
            self._user_visible[label_id] = True
        self.soloModeChanged.emit(False, "")
        self.allVisibilityChanged.emit()
    
    def hide_all(self):
        """Hide all labels."""
        for label_id in self._all_labels:
            self._user_visible[label_id] = False
        self.allVisibilityChanged.emit()
    
    # ---- Persistence ----
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize visibility settings to a dictionary."""
        return {
            "list_filter_mode": self._list_filter_mode.value,
            "view_hidden_states": [s.value for s in self._view_hidden_states],
            "user_visible": self._user_visible.copy(),
            "solo_mode": self._solo_mode,
            "solo_label": self._solo_label,
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Deserialize visibility settings from a dictionary."""
        if not data:
            return
        
        # List filter mode
        filter_value = data.get("list_filter_mode", "all")
        try:
            self._list_filter_mode = LabelFilterMode(filter_value)
        except ValueError:
            self._list_filter_mode = LabelFilterMode.ALL
        
        # View hidden states
        hidden_values = data.get("view_hidden_states", ["verified"])
        self._view_hidden_states = set()
        for v in hidden_values:
            try:
                self._view_hidden_states.add(LabelState(v))
            except ValueError:
                pass
        
        # User visible flags
        self._user_visible = data.get("user_visible", {})
        
        # Solo mode
        self._solo_mode = data.get("solo_mode", False)
        self._solo_label = data.get("solo_label", "")
        
        # Update all labels set from user_visible keys
        self._all_labels = set(self._user_visible.keys())
    
    def reset(self):
        """Reset to default state."""
        self._list_filter_mode = LabelFilterMode.ALL
        self._view_hidden_states = {LabelState.VERIFIED}
        self._user_visible.clear()
        self._all_labels.clear()
        self._solo_mode = False
        self._solo_label = ""
    
    # ---- Internal Helpers ----
    
    def _get_label_state(self, label_id: str) -> Optional[LabelState]:
        """Get the state of a label from metadata store."""
        if self._metadata_store:
            return self._metadata_store.get_state(str(label_id))
        return None
    
    def _emit_visibility_changes_for_state(self, state: LabelState):
        """Emit visibility change signals for all labels with the given state."""
        if not self._metadata_store:
            return
        
        labels = self._metadata_store.get_labels_by_state(state)
        for label_id in labels:
            effective = self.get_effective_visible(label_id)
            self.effectiveVisibilityChanged.emit(label_id, effective)
