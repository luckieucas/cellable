# tests/labelme_tests/test_label_state.py
# -*- coding: utf-8 -*-
"""
Unit tests for the label lifecycle state management module.

Tests cover:
- LabelState enum
- LabelOrigin enum
- LabelMetadata class
- LabelMetadataStore class
- State transitions: PROPOSED -> EDITED -> VERIFIED
- Revert and commit operations
- Merge and split operations
- Persistence (save/load)
- Undo/redo functionality
"""

import pytest
import numpy as np
import tempfile
import os

from labelme.label_state import (
    LabelState,
    LabelOrigin,
    LabelMetadata,
    LabelMetadataStore,
    encode_mask_rle,
    decode_mask_rle,
)


class TestLabelStateEnum:
    """Tests for LabelState enum."""
    
    def test_state_values(self):
        """Test that state enum has expected values."""
        assert LabelState.PROPOSED.value == "proposed"
        assert LabelState.EDITED.value == "edited"
        assert LabelState.VERIFIED.value == "verified"
    
    def test_state_str(self):
        """Test string representation of states."""
        assert str(LabelState.PROPOSED) == "proposed"
        assert str(LabelState.EDITED) == "edited"
        assert str(LabelState.VERIFIED) == "verified"


class TestLabelOriginEnum:
    """Tests for LabelOrigin enum."""
    
    def test_origin_values(self):
        """Test that origin enum has expected values."""
        assert LabelOrigin.AI.value == "ai"
        assert LabelOrigin.WATERSHED.value == "watershed"
        assert LabelOrigin.INTERPOLATION.value == "interpolation"
        assert LabelOrigin.MANUAL.value == "manual"
        assert LabelOrigin.TRACKING.value == "tracking"
        assert LabelOrigin.UNKNOWN.value == "unknown"


class TestMaskRLE:
    """Tests for RLE encoding/decoding of masks."""
    
    def test_encode_decode_simple(self):
        """Test RLE encoding and decoding of a simple mask."""
        mask = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint8)
        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)
    
    def test_encode_decode_3d(self):
        """Test RLE encoding and decoding of a 3D mask."""
        mask = np.random.randint(0, 2, size=(10, 20, 30), dtype=np.uint8)
        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)
    
    def test_encode_decode_all_zeros(self):
        """Test RLE encoding and decoding of an all-zeros mask."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)
    
    def test_encode_decode_all_ones(self):
        """Test RLE encoding and decoding of an all-ones mask."""
        mask = np.ones((5, 5), dtype=np.uint8)
        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)
    
    def test_encode_empty_mask(self):
        """Test RLE encoding of None mask."""
        encoded = encode_mask_rle(None)
        assert encoded == ""
    
    def test_decode_empty_string(self):
        """Test RLE decoding of empty string."""
        decoded = decode_mask_rle("", (5, 5))
        assert decoded is None


class TestLabelMetadata:
    """Tests for LabelMetadata class."""
    
    def test_creation_with_defaults(self):
        """Test creating metadata with default values."""
        meta = LabelMetadata(label_id="1")
        assert meta.label_id == "1"
        assert meta.state == LabelState.PROPOSED
        assert meta.origin == LabelOrigin.UNKNOWN
        assert meta.created_at != ""
        assert meta.last_modified_at != ""
    
    def test_creation_with_values(self):
        """Test creating metadata with specific values."""
        meta = LabelMetadata(
            label_id="42",
            state=LabelState.VERIFIED,
            origin=LabelOrigin.AI,
        )
        assert meta.label_id == "42"
        assert meta.state == LabelState.VERIFIED
        assert meta.origin == LabelOrigin.AI
    
    def test_mark_edited(self):
        """Test marking label as edited."""
        meta = LabelMetadata(label_id="1", state=LabelState.PROPOSED)
        old_modified = meta.last_modified_at
        meta.mark_edited()
        assert meta.state == LabelState.EDITED
        # Timestamp should be updated
        assert meta.last_modified_at >= old_modified
    
    def test_mark_verified(self):
        """Test marking label as verified."""
        meta = LabelMetadata(label_id="1", state=LabelState.EDITED)
        meta.mark_verified()
        assert meta.state == LabelState.VERIFIED
        assert meta.verified_at != ""
    
    def test_mark_proposed(self):
        """Test marking label back to proposed."""
        meta = LabelMetadata(label_id="1", state=LabelState.VERIFIED)
        meta.mark_proposed()
        assert meta.state == LabelState.PROPOSED
        assert meta.verified_at == ""
    
    def test_proposed_snapshot(self):
        """Test storing and retrieving proposed snapshot."""
        meta = LabelMetadata(label_id="1")
        mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        
        meta.set_proposed_snapshot(mask)
        assert meta.has_proposed_snapshot()
        
        retrieved = meta.get_proposed_snapshot()
        np.testing.assert_array_equal(mask, retrieved)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        meta = LabelMetadata(
            label_id="42",
            state=LabelState.EDITED,
            origin=LabelOrigin.WATERSHED,
            notes="Test note",
        )
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        meta.set_proposed_snapshot(mask)
        
        data = meta.to_dict()
        restored = LabelMetadata.from_dict(data)
        
        assert restored.label_id == meta.label_id
        assert restored.state == meta.state
        assert restored.origin == meta.origin
        assert restored.notes == meta.notes
        
        restored_mask = restored.get_proposed_snapshot()
        np.testing.assert_array_equal(mask, restored_mask)


class TestLabelMetadataStore:
    """Tests for LabelMetadataStore class."""
    
    def test_get_or_create(self):
        """Test getting or creating label metadata."""
        store = LabelMetadataStore()
        
        # Should create new
        meta1 = store.get_or_create("1", origin=LabelOrigin.AI)
        assert meta1.label_id == "1"
        assert meta1.origin == LabelOrigin.AI
        
        # Should return existing
        meta2 = store.get_or_create("1")
        assert meta1 is meta2
    
    def test_get_nonexistent(self):
        """Test getting nonexistent label."""
        store = LabelMetadataStore()
        assert store.get("999") is None
    
    def test_remove_and_restore(self):
        """Test removing and restoring labels."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        
        removed = store.remove("1")
        assert removed is not None
        assert store.get("1") is None
        
        store.restore("1", removed)
        assert store.get("1") is not None
    
    def test_get_labels_by_state(self):
        """Test filtering labels by state."""
        store = LabelMetadataStore()
        store.get_or_create("1")  # PROPOSED by default
        store.get_or_create("2").mark_edited()
        store.get_or_create("3").mark_verified()
        store.get_or_create("4")  # PROPOSED
        
        proposed = store.get_labels_by_state(LabelState.PROPOSED)
        edited = store.get_labels_by_state(LabelState.EDITED)
        verified = store.get_labels_by_state(LabelState.VERIFIED)
        
        assert set(proposed) == {"1", "4"}
        assert set(edited) == {"2"}
        assert set(verified) == {"3"}
    
    def test_verify_label(self):
        """Test verifying a label."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        
        store.verify_label("1")
        assert store.get_state("1") == LabelState.VERIFIED
    
    def test_unverify_label(self):
        """Test unverifying a label."""
        store = LabelMetadataStore()
        store.get_or_create("1").mark_verified()
        
        store.unverify_label("1")
        assert store.get_state("1") == LabelState.EDITED
    
    def test_create_from_auto_segmentation(self):
        """Test creating label from auto-segmentation."""
        store = LabelMetadataStore()
        mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        
        meta = store.create_from_auto_segmentation("5", mask, LabelOrigin.AI)
        
        assert meta.state == LabelState.PROPOSED
        assert meta.origin == LabelOrigin.AI
        assert meta.has_proposed_snapshot()
    
    def test_revert_to_proposed(self):
        """Test reverting to proposed state."""
        store = LabelMetadataStore()
        mask = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        
        store.create_from_auto_segmentation("1", mask, LabelOrigin.AI)
        store.mark_edited("1")
        
        assert store.get_state("1") == LabelState.EDITED
        
        # Revert
        restored_mask = store.revert_to_proposed("1")
        
        assert store.get_state("1") == LabelState.PROPOSED
        np.testing.assert_array_equal(mask, restored_mask)
    
    def test_can_revert(self):
        """Test checking if label can be reverted."""
        store = LabelMetadataStore()
        
        # Label without snapshot
        store.get_or_create("1")
        assert not store.can_revert("1")
        
        # Label with snapshot
        mask = np.array([[1]], dtype=np.uint8)
        store.create_from_auto_segmentation("2", mask, LabelOrigin.AI)
        assert store.can_revert("2")
    
    def test_commit(self):
        """Test committing changes."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        store.get_or_create("2")
        
        working_mask = np.zeros((10, 10), dtype=np.uint16)
        revision = store.commit(working_mask)
        
        assert revision == 1
        assert store.get_commit_revision() == 1
        
        # Second commit
        revision2 = store.commit(working_mask)
        assert revision2 == 2
    
    def test_handle_merge(self):
        """Test handling merge operation."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        store.get_or_create("2")
        
        target_mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        store.handle_merge(["1"], "2", target_mask)
        
        # Source label should be removed
        assert store.get("1") is None
        
        # Target label should be EDITED with source tracking
        target_meta = store.get("2")
        assert target_meta.state == LabelState.EDITED
        assert "1" in target_meta.source_labels
    
    def test_handle_split(self):
        """Test handling split operation."""
        store = LabelMetadataStore()
        store.get_or_create("10", origin=LabelOrigin.AI)
        
        child_masks = {
            "11": np.array([[1, 0]], dtype=np.uint8),
            "12": np.array([[0, 1]], dtype=np.uint8),
        }
        store.handle_split("10", ["11", "12"], child_masks)
        
        # Parent should be removed
        assert store.get("10") is None
        
        # Children should exist and track parent
        child1 = store.get("11")
        child2 = store.get("12")
        assert child1.state == LabelState.EDITED
        assert child2.state == LabelState.EDITED
        assert child1.parent_label == "10"
        assert child2.parent_label == "10"
        assert child1.origin == LabelOrigin.AI  # Inherited from parent
    
    def test_undo_redo(self):
        """Test undo/redo functionality."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        
        # Make a change that can be undone
        store.set_state("1", LabelState.VERIFIED, push_undo=True)
        assert store.get_state("1") == LabelState.VERIFIED
        
        # Undo
        assert store.undo()
        assert store.get_state("1") == LabelState.PROPOSED
        
        # Redo
        assert store.redo()
        assert store.get_state("1") == LabelState.VERIFIED
    
    def test_save_and_load(self):
        """Test saving and loading metadata."""
        store = LabelMetadataStore()
        mask = np.array([[1, 0, 1]], dtype=np.uint8)
        store.create_from_auto_segmentation("1", mask, LabelOrigin.AI)
        store.verify_label("1")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metadata.json")
            store.save(filepath)
            
            # Load into new store
            new_store = LabelMetadataStore()
            assert new_store.load(filepath)
            
            # Verify loaded data
            loaded_meta = new_store.get("1")
            assert loaded_meta.state == LabelState.VERIFIED
            assert loaded_meta.origin == LabelOrigin.AI
            
            loaded_mask = loaded_meta.get_proposed_snapshot()
            np.testing.assert_array_equal(mask, loaded_mask)
    
    def test_get_stats(self):
        """Test getting label statistics."""
        store = LabelMetadataStore()
        store.get_or_create("1")  # PROPOSED
        store.get_or_create("2")  # PROPOSED
        store.get_or_create("3").mark_edited()
        store.get_or_create("4").mark_verified()
        
        stats = store.get_stats()
        
        assert stats["total"] == 4
        assert stats["proposed"] == 2
        assert stats["edited"] == 1
        assert stats["verified"] == 1
    
    def test_clear(self):
        """Test clearing the store."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        store.get_or_create("2")
        
        store.clear()
        
        assert len(store) == 0
        assert store.get_commit_revision() == 0
    
    def test_contains_and_len(self):
        """Test __contains__ and __len__."""
        store = LabelMetadataStore()
        store.get_or_create("1")
        
        assert "1" in store
        assert "2" not in store
        assert len(store) == 1
    
    def test_get_sidecar_path(self):
        """Test getting sidecar path."""
        assert LabelMetadataStore.get_sidecar_path("/path/to/mask.tif") == "/path/to/mask_metadata.json"
        assert LabelMetadataStore.get_sidecar_path("/path/to/mask.nii.gz") == "/path/to/mask_metadata.json"
        assert LabelMetadataStore.get_sidecar_path("/path/to/mask.nii") == "/path/to/mask_metadata.json"


class TestStateTransitions:
    """Integration tests for full state transition workflows."""
    
    def test_full_workflow_proposed_to_verified(self):
        """Test complete workflow: PROPOSED -> EDITED -> VERIFIED."""
        store = LabelMetadataStore()
        mask = np.ones((5, 5), dtype=np.uint8)
        
        # Auto-segmentation creates PROPOSED label
        store.create_from_auto_segmentation("1", mask, LabelOrigin.AI)
        assert store.get_state("1") == LabelState.PROPOSED
        
        # User edits the label
        store.mark_edited("1")
        assert store.get_state("1") == LabelState.EDITED
        
        # User verifies the label
        store.verify_label("1")
        assert store.get_state("1") == LabelState.VERIFIED
    
    def test_revert_after_edit(self):
        """Test reverting an edited label to proposed state."""
        store = LabelMetadataStore()
        original_mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        
        # Create from auto-seg
        store.create_from_auto_segmentation("1", original_mask, LabelOrigin.WATERSHED)
        
        # Edit the label (simulating user modification)
        store.mark_edited("1")
        assert store.get_state("1") == LabelState.EDITED
        
        # Revert to proposed
        restored = store.revert_to_proposed("1")
        assert store.get_state("1") == LabelState.PROPOSED
        np.testing.assert_array_equal(original_mask, restored)
    
    def test_verify_then_edit_requires_reverify(self):
        """Test that editing a verified label puts it back to EDITED."""
        store = LabelMetadataStore()
        mask = np.ones((3, 3), dtype=np.uint8)
        
        store.create_from_auto_segmentation("1", mask, LabelOrigin.AI)
        store.verify_label("1")
        assert store.get_state("1") == LabelState.VERIFIED
        
        # Re-edit the label
        store.mark_edited("1")
        assert store.get_state("1") == LabelState.EDITED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
