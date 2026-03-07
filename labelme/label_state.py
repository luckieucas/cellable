# labelme/label_state.py
# -*- coding: utf-8 -*-
"""
Label lifecycle state management for annotation workflow.

States:
- PROPOSED: Labels produced by AI/watershed/auto-seg (not yet verified by user)
- EDITED: User has modified the label
- VERIFIED: User has confirmed the label is correct

This module provides:
- LabelState enum for state representation
- LabelOrigin enum for tracking how labels were created
- LabelMetadata class for per-label metadata
- LabelMetadataStore for managing all label metadata with persistence
"""

import json
import os
import numpy as np
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import base64
import zlib


class LabelState(Enum):
    """State of a label in the annotation lifecycle."""
    PROPOSED = "proposed"    # Auto-generated, not yet reviewed
    EDITED = "edited"        # User has made modifications
    VERIFIED = "verified"    # User has confirmed correctness
    
    def __str__(self):
        return self.value


class LabelOrigin(Enum):
    """Origin/source of how a label was created."""
    AI = "ai"                      # Created by AI segmentation (SAM, CellPose, etc.)
    WATERSHED = "watershed"         # Created by watershed algorithm
    INTERPOLATION = "interpolation" # Created by interpolation between slices
    MANUAL = "manual"               # Created manually by user
    TRACKING = "tracking"           # Created by tracking across slices
    UNKNOWN = "unknown"             # Unknown origin (legacy data)
    
    def __str__(self):
        return self.value


def encode_mask_rle(mask: np.ndarray) -> str:
    """
    Encode a binary mask using Run-Length Encoding (RLE).
    Returns a base64-encoded compressed string.
    
    Args:
        mask: 2D or 3D numpy array (will be flattened)
    
    Returns:
        Base64-encoded compressed RLE string
    """
    if mask is None:
        return ""
    
    # Flatten mask and convert to uint8
    flat = mask.flatten().astype(np.uint8)
    
    # RLE encoding
    if len(flat) == 0:
        return ""
    
    # Find runs
    runs = []
    current_val = flat[0]
    current_count = 1
    
    for val in flat[1:]:
        if val == current_val:
            current_count += 1
        else:
            runs.append((current_val, current_count))
            current_val = val
            current_count = 1
    runs.append((current_val, current_count))
    
    # Encode as bytes: [val, count_bytes...]
    # Use variable-length encoding for counts
    encoded = []
    for val, count in runs:
        encoded.append(val)
        # Encode count as variable-length integer
        while count >= 128:
            encoded.append((count & 0x7F) | 0x80)
            count >>= 7
        encoded.append(count)
    
    # Compress and base64 encode
    compressed = zlib.compress(bytes(encoded))
    return base64.b64encode(compressed).decode('ascii')


def decode_mask_rle(encoded: str, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """
    Decode an RLE-encoded mask back to a numpy array.
    
    Args:
        encoded: Base64-encoded compressed RLE string
        shape: Original shape of the mask
    
    Returns:
        Decoded numpy array or None if decoding fails
    """
    if not encoded:
        return None
    
    try:
        # Decompress
        compressed = base64.b64decode(encoded.encode('ascii'))
        encoded_bytes = zlib.decompress(compressed)
        
        # Decode RLE
        decoded = []
        i = 0
        while i < len(encoded_bytes):
            val = encoded_bytes[i]
            i += 1
            
            # Decode variable-length count
            count = 0
            shift = 0
            while i < len(encoded_bytes):
                byte = encoded_bytes[i]
                i += 1
                count |= (byte & 0x7F) << shift
                if (byte & 0x80) == 0:
                    break
                shift += 7
            
            decoded.extend([val] * count)
        
        # Reshape
        total_size = np.prod(shape)
        if len(decoded) != total_size:
            return None
        
        return np.array(decoded, dtype=np.uint8).reshape(shape)
    except Exception:
        return None


@dataclass
class LabelMetadata:
    """Metadata for a single label in the annotation workflow."""
    label_id: str
    state: LabelState = LabelState.PROPOSED
    origin: LabelOrigin = LabelOrigin.UNKNOWN
    
    # Timestamps
    created_at: str = ""
    last_modified_at: str = ""
    verified_at: str = ""
    
    # Snapshot data (stored as RLE-encoded strings)
    proposed_snapshot_rle: str = ""
    proposed_snapshot_shape: Tuple[int, ...] = ()
    
    # For merge/split tracking
    source_labels: List[str] = field(default_factory=list)  # For merged labels
    parent_label: str = ""  # For split labels
    
    # Commit tracking
    last_commit_revision: int = 0
    
    # Additional metadata
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_modified_at:
            self.last_modified_at = self.created_at
    
    def set_proposed_snapshot(self, mask: np.ndarray):
        """Store the proposed snapshot from a mask array."""
        if mask is not None:
            self.proposed_snapshot_rle = encode_mask_rle(mask)
            self.proposed_snapshot_shape = mask.shape
        else:
            self.proposed_snapshot_rle = ""
            self.proposed_snapshot_shape = ()
    
    def get_proposed_snapshot(self) -> Optional[np.ndarray]:
        """Retrieve the proposed snapshot as a mask array."""
        if not self.proposed_snapshot_rle or not self.proposed_snapshot_shape:
            return None
        return decode_mask_rle(self.proposed_snapshot_rle, self.proposed_snapshot_shape)
    
    def has_proposed_snapshot(self) -> bool:
        """Check if this label has a proposed snapshot stored."""
        return bool(self.proposed_snapshot_rle and self.proposed_snapshot_shape)
    
    def mark_edited(self):
        """Mark label as edited and update timestamp."""
        if self.state == LabelState.VERIFIED:
            # Re-editing a verified label puts it back to edited
            pass
        self.state = LabelState.EDITED
        self.last_modified_at = datetime.now().isoformat()
    
    def mark_verified(self):
        """Mark label as verified."""
        self.state = LabelState.VERIFIED
        self.verified_at = datetime.now().isoformat()
        self.last_modified_at = self.verified_at
    
    def mark_proposed(self):
        """Mark label as proposed (e.g., after revert)."""
        self.state = LabelState.PROPOSED
        self.last_modified_at = datetime.now().isoformat()
        self.verified_at = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label_id": self.label_id,
            "state": self.state.value,
            "origin": self.origin.value,
            "created_at": self.created_at,
            "last_modified_at": self.last_modified_at,
            "verified_at": self.verified_at,
            "proposed_snapshot_rle": self.proposed_snapshot_rle,
            "proposed_snapshot_shape": list(self.proposed_snapshot_shape),
            "source_labels": self.source_labels,
            "parent_label": self.parent_label,
            "last_commit_revision": self.last_commit_revision,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelMetadata":
        """Create from dictionary (JSON deserialization)."""
        state_value = str(data.get("state", "proposed")).lower()
        # Backward compatibility: some legacy files used "manual" as a state.
        # Map it to EDITED in the current lifecycle model.
        if state_value == "manual":
            state_value = LabelState.EDITED.value

        if state_value not in {s.value for s in LabelState}:
            state_value = LabelState.PROPOSED.value

        origin_value = str(data.get("origin", "unknown")).lower()
        if origin_value not in {o.value for o in LabelOrigin}:
            origin_value = LabelOrigin.UNKNOWN.value

        return cls(
            label_id=data.get("label_id", ""),
            state=LabelState(state_value),
            origin=LabelOrigin(origin_value),
            created_at=data.get("created_at", ""),
            last_modified_at=data.get("last_modified_at", ""),
            verified_at=data.get("verified_at", ""),
            proposed_snapshot_rle=data.get("proposed_snapshot_rle", ""),
            proposed_snapshot_shape=tuple(data.get("proposed_snapshot_shape", [])),
            source_labels=data.get("source_labels", []),
            parent_label=data.get("parent_label", ""),
            last_commit_revision=data.get("last_commit_revision", 0),
            notes=data.get("notes", ""),
        )


class LabelMetadataStore:
    """
    Manages metadata for all labels in the annotation workflow.
    
    Features:
    - Per-label state tracking (PROPOSED, EDITED, VERIFIED)
    - Proposed mask snapshots for reverting
    - Commit/revert functionality
    - Persistence to JSON sidecar file
    """
    
    VERSION = 1
    
    def __init__(self):
        self._labels: Dict[str, LabelMetadata] = {}
        self._commit_revision: int = 0
        self._final_mask: Optional[np.ndarray] = None
        self._undo_stack: List[Tuple[str, LabelMetadata, str]] = []  # (action, metadata, label_id)
        self._redo_stack: List[Tuple[str, LabelMetadata, str]] = []
        self._undo_limit = 50
    
    def get(self, label_id: str) -> Optional[LabelMetadata]:
        """Get metadata for a label, or None if not found."""
        return self._labels.get(str(label_id))
    
    def get_or_create(self, label_id: str, 
                      origin: LabelOrigin = LabelOrigin.UNKNOWN) -> LabelMetadata:
        """Get metadata for a label, creating it if it doesn't exist."""
        label_id = str(label_id)
        if label_id not in self._labels:
            self._labels[label_id] = LabelMetadata(
                label_id=label_id,
                # "MANUAL" is an origin, not a state. Manual labels should start as EDITED.
                state=LabelState.EDITED if origin == LabelOrigin.MANUAL else LabelState.PROPOSED,
                origin=origin,
            )
        return self._labels[label_id]
    
    def set(self, label_id: str, metadata: LabelMetadata):
        """Set metadata for a label."""
        self._labels[str(label_id)] = metadata
    
    def remove(self, label_id: str, push_undo: bool = True) -> Optional[LabelMetadata]:
        """Remove a label and return its metadata (for undo)."""
        label_id = str(label_id)
        if label_id in self._labels:
            metadata = self._labels.pop(label_id)
            if push_undo:
                self._push_undo("remove", metadata, label_id)
            return metadata
        return None
    
    def restore(self, label_id: str, metadata: LabelMetadata, push_undo: bool = True):
        """Restore a previously removed label."""
        label_id = str(label_id)
        if push_undo and label_id in self._labels:
            self._push_undo("restore", self._labels[label_id], label_id)
        self._labels[label_id] = metadata
    
    def get_all_labels(self) -> List[str]:
        """Get all label IDs."""
        return list(self._labels.keys())
    
    def get_labels_by_state(self, state: LabelState) -> List[str]:
        """Get all labels with a specific state."""
        return [lid for lid, meta in self._labels.items() if meta.state == state]
    
    def get_state(self, label_id: str) -> Optional[LabelState]:
        """Get the state of a label."""
        meta = self.get(str(label_id))
        return meta.state if meta else None
    
    def set_state(self, label_id: str, state: LabelState, push_undo: bool = True):
        """Set the state of a label."""
        label_id = str(label_id)
        meta = self.get_or_create(label_id)
        if push_undo:
            self._push_undo("set_state", LabelMetadata.from_dict(meta.to_dict()), label_id)
        
        if state == LabelState.PROPOSED:
            meta.mark_proposed()
        elif state == LabelState.EDITED:
            meta.mark_edited()
        elif state == LabelState.VERIFIED:
            meta.mark_verified()
    
    def mark_edited(self, label_id: str, push_undo: bool = False):
        """Mark a label as edited (typically called automatically on mask changes)."""
        label_id = str(label_id)
        meta = self.get(label_id)
        if meta and meta.state != LabelState.EDITED:
            if push_undo:
                self._push_undo("mark_edited", LabelMetadata.from_dict(meta.to_dict()), label_id)
            meta.mark_edited()
    
    def mark_labels_edited(self, label_ids: List[str], push_undo: bool = False):
        """Mark multiple labels as edited."""
        for label_id in label_ids:
            self.mark_edited(label_id, push_undo=push_undo)
    
    def create_from_auto_segmentation(self, label_id: str, mask: np.ndarray, 
                                       origin: LabelOrigin) -> LabelMetadata:
        """
        Create a new label from auto-segmentation with PROPOSED state.
        Stores the proposed snapshot for potential revert.
        """
        label_id = str(label_id)
        metadata = LabelMetadata(
            label_id=label_id,
            state=LabelState.PROPOSED,
            origin=origin,
        )
        metadata.set_proposed_snapshot(mask)
        self._labels[label_id] = metadata
        return metadata
    
    def verify_label(self, label_id: str, push_undo: bool = True):
        """Mark a label as verified by the user."""
        self.set_state(str(label_id), LabelState.VERIFIED, push_undo=push_undo)
    
    def unverify_label(self, label_id: str, push_undo: bool = True):
        """Unverify a label (VERIFIED -> EDITED)."""
        label_id = str(label_id)
        meta = self.get(label_id)
        if meta and meta.state == LabelState.VERIFIED:
            self.set_state(label_id, LabelState.EDITED, push_undo=push_undo)
    
    def can_revert(self, label_id: str) -> bool:
        """Check if a label can be reverted to its proposed state."""
        meta = self.get(str(label_id))
        return meta is not None and meta.has_proposed_snapshot()
    
    def get_proposed_snapshot(self, label_id: str) -> Optional[np.ndarray]:
        """Get the proposed snapshot for a label (for revert operation)."""
        meta = self.get(str(label_id))
        if meta:
            return meta.get_proposed_snapshot()
        return None
    
    def revert_to_proposed(self, label_id: str, push_undo: bool = True) -> Optional[np.ndarray]:
        """
        Revert a label to its proposed state.
        Returns the proposed snapshot mask if available.
        """
        label_id = str(label_id)
        meta = self.get(label_id)
        if meta and meta.has_proposed_snapshot():
            if push_undo:
                self._push_undo("revert", LabelMetadata.from_dict(meta.to_dict()), label_id)
            meta.mark_proposed()
            return meta.get_proposed_snapshot()
        return None
    
    def commit(self, working_mask: np.ndarray) -> int:
        """
        Commit the current working mask to the final mask store.
        Returns the new commit revision number.
        
        Note: This does NOT erase proposed snapshots.
        """
        self._commit_revision += 1
        self._final_mask = working_mask.copy()
        
        # Update commit revision for all labels
        for label_id in self._labels:
            self._labels[label_id].last_commit_revision = self._commit_revision
        
        return self._commit_revision
    
    def get_final_mask(self) -> Optional[np.ndarray]:
        """Get the last committed final mask."""
        return self._final_mask
    
    def get_commit_revision(self) -> int:
        """Get the current commit revision number."""
        return self._commit_revision
    
    # ----- Undo/Redo Support -----
    
    def _push_undo(self, action: str, metadata: LabelMetadata, label_id: str):
        """Push an action to the undo stack."""
        self._undo_stack.append((action, metadata, label_id))
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
    
    def undo(self) -> bool:
        """Undo the last metadata operation. Returns True if successful."""
        if not self._undo_stack:
            return False
        
        action, metadata, label_id = self._undo_stack.pop()
        
        # Store current state for redo
        current_meta = self.get(label_id)
        if current_meta:
            self._redo_stack.append((action, LabelMetadata.from_dict(current_meta.to_dict()), label_id))
        elif action == "remove":
            self._redo_stack.append(("restore", metadata, label_id))
        
        # Restore previous state
        if action == "remove":
            self._labels[label_id] = metadata
        elif action == "restore":
            self._labels.pop(label_id, None)
        else:
            self._labels[label_id] = metadata
        
        return True
    
    def redo(self) -> bool:
        """Redo the last undone operation. Returns True if successful."""
        if not self._redo_stack:
            return False
        
        action, metadata, label_id = self._redo_stack.pop()
        
        # Store current state for undo
        current_meta = self.get(label_id)
        if current_meta:
            self._undo_stack.append((action, LabelMetadata.from_dict(current_meta.to_dict()), label_id))
        elif action == "restore":
            self._undo_stack.append(("remove", metadata, label_id))
        
        # Apply redo state
        if action == "remove":
            self._labels.pop(label_id, None)
        elif action == "restore":
            self._labels[label_id] = metadata
        else:
            self._labels[label_id] = metadata
        
        return True
    
    def clear_undo_history(self):
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
    
    # ----- Merge/Split Support -----
    
    def handle_merge(self, source_labels: List[str], target_label: str, 
                     target_mask: np.ndarray, push_undo: bool = True):
        """
        Handle a merge operation where multiple labels are merged into one.
        The target label gets EDITED state and tracks source labels.
        """
        target_label = str(target_label)
        source_labels = [str(l) for l in source_labels]
        
        # Create target metadata
        target_meta = LabelMetadata(
            label_id=target_label,
            state=LabelState.EDITED,
            origin=LabelOrigin.MANUAL,
            source_labels=source_labels,
        )
        target_meta.set_proposed_snapshot(target_mask)
        
        if push_undo:
            # Save state for undo
            for src in source_labels:
                if src in self._labels:
                    self._push_undo("merge_source", self._labels[src], src)
        
        # Remove source labels (but keep their snapshots in the target's metadata)
        for src in source_labels:
            self._labels.pop(src, None)
        
        self._labels[target_label] = target_meta
    
    def handle_split(self, parent_label: str, child_labels: List[str], 
                     child_masks: Dict[str, np.ndarray], push_undo: bool = True):
        """
        Handle a split operation where one label is split into multiple.
        Child labels get EDITED state and track parent label.
        """
        parent_label = str(parent_label)
        child_labels = [str(l) for l in child_labels]
        
        if push_undo and parent_label in self._labels:
            self._push_undo("split_parent", self._labels[parent_label], parent_label)
        
        parent_origin = self._labels.get(parent_label, LabelMetadata(parent_label)).origin
        
        # Create child metadata
        for child_label in child_labels:
            child_meta = LabelMetadata(
                label_id=child_label,
                state=LabelState.EDITED,
                origin=parent_origin,
                parent_label=parent_label,
            )
            if child_label in child_masks:
                child_meta.set_proposed_snapshot(child_masks[child_label])
            self._labels[child_label] = child_meta
        
        # Remove parent label
        self._labels.pop(parent_label, None)
    
    # ----- Persistence -----
    
    def save(self, filepath: str):
        """Save metadata to a JSON sidecar file."""
        data = {
            "version": self.VERSION,
            "commit_revision": self._commit_revision,
            "labels": {lid: meta.to_dict() for lid, meta in self._labels.items()},
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str) -> bool:
        """
        Load metadata from a JSON sidecar file.
        Returns True if successful.
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            version = data.get("version", 1)
            self._commit_revision = data.get("commit_revision", 0)
            
            self._labels.clear()
            for lid, meta_dict in data.get("labels", {}).items():
                self._labels[lid] = LabelMetadata.from_dict(meta_dict)
            
            return True
        except Exception as e:
            print(f"Error loading label metadata: {e}")
            return False
    
    @staticmethod
    def get_sidecar_path(mask_filepath: str) -> str:
        """Get the sidecar JSON path for a given mask file."""
        base, ext = os.path.splitext(mask_filepath)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
        return base + "_metadata.json"
    
    def clear(self):
        """Clear all metadata."""
        self._labels.clear()
        self._commit_revision = 0
        self._final_mask = None
        self._undo_stack.clear()
        self._redo_stack.clear()
    
    # ----- Statistics -----
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about label states."""
        stats = {
            "total": len(self._labels),
            "proposed": 0,
            "edited": 0,
            "verified": 0,
        }
        for meta in self._labels.values():
            if meta.state == LabelState.PROPOSED:
                stats["proposed"] += 1
            elif meta.state == LabelState.EDITED:
                stats["edited"] += 1
            elif meta.state == LabelState.VERIFIED:
                stats["verified"] += 1
        return stats
    
    def __len__(self) -> int:
        return len(self._labels)
    
    def __contains__(self, label_id: str) -> bool:
        return str(label_id) in self._labels
    
    def __iter__(self):
        return iter(self._labels.items())
