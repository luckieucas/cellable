# labelme/app.py

# -*- coding: utf-8 -*-
import collections
import threading
import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser
import zlib
import tifffile as tiff
import SimpleITK as sitk
import json
import cc3d
import natsort
import scipy.ndimage
from skimage.segmentation import watershed 
from scipy import ndimage as ndi
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt
import queue


import imgviz
import natsort
import numpy as np
from scipy.ndimage import measurements
from scipy.spatial.distance import cdist

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QRegExp
from qtpy.QtGui import QRegExpValidator
import vtk
from vtk.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from labelme import PY2
from labelme import __appname__
from labelme import ai
from labelme.ai import MODELS
from labelme.config import get_config
from labelme.config import get_user_config_path
from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ShortcutSettingsDialog
from labelme.widgets import ShortcutSettingsWidget
from labelme.widgets import ZoomWidget
from labelme.utils import compute_tiff_sam_feature, compute_points_from_mask
from labelme.label_state import (
    LabelState, LabelOrigin, LabelMetadata, LabelMetadataStore
)
from labelme.label_visibility import (
    LabelFilterMode, LabelVisibilityManager
)
from PyQt5.QtWidgets import QSplitter, QLineEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QWidgetAction, QLineEdit, QPushButton, QLabel,  QSizePolicy

try:
    from . import utils
except:
    import utils



LABEL_COLORMAP = imgviz.label_colormap()
OFFSET_LABEL = 1000
MAX_LABEL = 2000

# Maximum number of slices to keep undo/mask history for (LRU eviction to prevent slowdown)
MAX_SLICES_HISTORY = 50

# Scroll throttle: min ms between slice changes during wheel scroll (keeps scroll speed constant)
SLICE_SCROLL_THROTTLE_MS = 40

# Max number of slice pixmaps to cache (bounded to prevent memory growth and slowdown over time)
MAX_SLICE_PIXMAP_CACHE = 256

# Max number of slice mask shapes to cache (speeds up revisiting slices)
MAX_SLICE_SHAPE_CACHE = 256

# Eagerly materializing mask shapes for every slice is expensive for dense
# instance volumes. Keep it opt-in; the normal path caches slices on demand.
PRECACHE_ALL_MASK_SHAPES_ON_OPEN = False

# Maximum merge undo/redo steps
MERGE_UNDO_LIMIT = 10

# Maximum watershed undo/redo steps (full-volume snapshots)
WATERSHED_UNDO_LIMIT = 10

# Periodic crash recovery for mask edits. This writes beside the real mask file
# and never replaces the user-visible mask until the user explicitly saves.
MASK_AUTOSAVE_INTERVAL_MS = 60000

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera


def process_mask(label, mask_data, slice_id):
    """
    Process a single label to create a mask shape.
    Uses np.where for bbox (avoids full 512x512 boolean array).
    """
    if label == 0:
        return None  # Skip the background
    rows, cols = np.where(mask_data == label)
    if rows.size == 0:
        return None
    y1, y2 = int(rows.min()), int(rows.max())
    x1, x2 = int(cols.min()), int(cols.max())
    mask_crop = (mask_data[y1 : y2 + 1, x1 : x2 + 1] == label)

    drawing_shape = Shape(
        label=str(label),
        shape_type="mask",
        description=f"Mask for label {label}",
        slice_id=slice_id,
    )
    drawing_shape.setShapeRefined(
        shape_type="mask",
        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
        point_labels=[1, 1],
        mask=mask_crop,
    )
    return drawing_shape


def _make_mask_shape_from_roi(label, y1, y2, x1, x2, roi_mask, slice_id):
    drawing_shape = Shape(
        label=str(label),
        shape_type="mask",
        description=f"Mask for label {label}",
        slice_id=slice_id,
    )
    drawing_shape.setShapeRefined(
        shape_type="mask",
        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
        point_labels=[1, 1],
        mask=roi_mask,
    )
    return drawing_shape


def _compute_shapes_from_mask_slice(mask_data, slice_id):
    """
    Build all mask shapes in a slice with one full pass over non-zero pixels.

    The old path called np.where(mask_data == label) once for every label,
    which scales poorly when a slice contains many labels.
    """
    rows, cols = np.nonzero(mask_data)
    if rows.size == 0:
        return []

    labels = mask_data[rows, cols]
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    rows = rows[order]
    cols = cols[order]

    change_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, labels.size]

    shapes = []
    for start, end in zip(starts, ends):
        label = int(labels[start])
        if label == 0:
            continue
        label_rows = rows[start:end]
        label_cols = cols[start:end]
        y1, y2 = int(label_rows.min()), int(label_rows.max())
        x1, x2 = int(label_cols.min()), int(label_cols.max())
        roi_mask = mask_data[y1 : y2 + 1, x1 : x2 + 1] == label
        shapes.append(
            _make_mask_shape_from_roi(label, y1, y2, x1, x2, roi_mask, slice_id)
        )
    return shapes


def _compute_shapes_for_slice(mask_volume, view_axis, slice_idx):
    """
    Compute mask shapes for a single slice. Thread-safe; used for pre-caching.
    Returns list of Shape objects.
    """
    idx = [slice(None)] * mask_volume.ndim
    idx[view_axis] = slice_idx
    mask_data = np.ascontiguousarray(mask_volume[tuple(idx)])
    return _compute_shapes_from_mask_slice(mask_data, slice_idx)


class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super().__init__()
        self.rotation_speed = 0.3  # Set rotation sensitivity; smaller value rotates slower
        self.zoom_speed = 0.5     # Set zoom sensitivity; smaller value zooms slower

    def Rotate(self):
        # Slow down rotation
        self.MotionFactor *= self.rotation_speed
        super().Rotate()

    def Dolly(self):
        # Slow down zoom
        self.MotionFactor *= self.zoom_speed
        super().Dolly()

def numpy_to_vtk_image(data: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=None):
    """
    Convert a 3D numpy array to vtkImageData more efficiently.

    Parameters:
        data (np.ndarray): 3D numpy array.
        spacing (tuple): Voxel spacing in (x, y, z) order. Default is (1.0, 1.0, 1.0).

    Returns:
        vtk.vtkImageData: Converted VTK image data.
    """
    # Ensure the numpy array is contiguous in memory
    data = np.ascontiguousarray(data)

    # Create a vtkImageData object
    vtk_image = vtk.vtkImageData()
    depth, height, width = data.shape
    vtk_image.SetDimensions(width, height, depth)
    
    # Set the spacing for the vtkImageData
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    if origin is not None:
        vtk_image.SetOrigin(origin[0], origin[1], origin[2])

    # allocate 16-bit unsigned scalars (1 component)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
    # Wrap the numpy array into a VTK array
    vtk_array = numpy_support.numpy_to_vtk(num_array=data.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)

    # Set the VTK array as the scalars for the vtkImageData
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image


def _compute_label_bboxes_3d(data, labels_to_render=None):
    """Return label bboxes from one pass over non-zero voxels."""
    if labels_to_render is None:
        zs, ys, xs = np.nonzero(data)
    else:
        zs, ys, xs = np.nonzero(np.isin(data, list(labels_to_render)))
    if zs.size == 0:
        return []

    labels = data[zs, ys, xs]
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    zs = zs[order]
    ys = ys[order]
    xs = xs[order]

    change_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, labels.size]

    bboxes = []
    for start, end in zip(starts, ends):
        label = int(labels[start])
        if label == 0:
            continue
        label_zs = zs[start:end]
        label_ys = ys[start:end]
        label_xs = xs[start:end]
        bboxes.append(
            (
                label,
                int(label_zs.min()),
                int(label_zs.max()),
                int(label_ys.min()),
                int(label_ys.max()),
                int(label_xs.min()),
                int(label_xs.max()),
                int(end - start),
            )
        )
    return bboxes

def process_label(label, data, smooth_iterations, label_colormap, spacing=(1.0, 1.0, 1.0)):
    """
    Process a single label: create iso-surface, smooth it, and return actor.
    
    Parameters:
        label: The label value to process.
        data: The 3D volume data.
        smooth_iterations: Number of smoothing iterations.
        label_colormap: Color map for labels.
        spacing: Voxel spacing in (x, y, z) order.
    """
    if label == 0:
        # Skip background (label 0)
        return None

    # Create a binary mask for the current label
    label_data = data.copy()
    label_data[label_data != label] = 0

    # Convert the binary mask to vtkImageData with spacing
    vtk_image = numpy_to_vtk_image(label_data, spacing=spacing)

    # Extract iso-surface using vtkMarchingCubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, label)
    marching_cubes.ComputeNormalsOn()
    marching_cubes.Update()

    # Optional: Smooth the extracted surface
    if smooth_iterations > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetRelaxationFactor(0.1)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        surface_output = smoother.GetOutput()
    else:
        surface_output = marching_cubes.GetOutput()

    # Create a mapper for the extracted or smoothed surface
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface_output)
    mapper.ScalarVisibilityOff()

    # Create an actor for the surface
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Assign a color to the actor based on the label
    color = [c / 255.0 for c in label_colormap[label % len(label_colormap)]]
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(1.0)  # Fully opaque

    # Attach the label as a property of the actor
    actor.label = label

    return actor


def process_label_roi(label_info, data, smooth_iterations, label_colormap, spacing):
    label, z1, z2, y1, y2, x1, x2, _count = label_info
    crop = data[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1]
    label_data = np.pad((crop == label).astype(data.dtype) * label, 1)
    origin = (
        (x1 - 1) * spacing[0],
        (y1 - 1) * spacing[1],
        (z1 - 1) * spacing[2],
    )

    vtk_image = numpy_to_vtk_image(label_data, spacing=spacing, origin=origin)
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, label)
    marching_cubes.ComputeNormalsOn()
    marching_cubes.Update()

    if smooth_iterations > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetRelaxationFactor(0.1)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        surface_output = smoother.GetOutput()
    else:
        surface_output = marching_cubes.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface_output)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    color = [c / 255.0 for c in label_colormap[label % len(label_colormap)]]
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(1.0)
    actor.label = label
    return actor


def _label_roi_checksum(label_info, data):
    label, z1, z2, y1, y2, x1, x2, _count = label_info
    crop = data[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1]
    label_mask = np.ascontiguousarray(crop == label)
    return zlib.adler32(label_mask.tobytes())

class VTKSurfaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the VTK RenderWindowInteractor for interactive 3D rendering
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout()
        layout.addWidget(self.vtkWidget)
        self.setLayout(layout)
        
        self.camera_initialized = False

        # Create the VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground([1.0,1.0, 1.0])  # White background

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # Initialize the interactor
        custom_style = CustomInteractorStyle()
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(custom_style)
        #self.highlight_actors = []  # List to store actors for highlighting
        self.crosshair_actors = []
        self._crosshair_sources = {}
        self._create_persistent_crosshair()
        self._axes_actor = None  # cache axes actor
        self._label_text_actor = None  # overlay for "Label: X / Y"
        self._cache_text_actor = None  # overlay for "Shapes: N | Slices: M"
        self._solo_text_actor = None  # overlay for "Solo: {label}"
        self._surface_actor_cache = {}

    def clear_surface_cache(self):
        self._surface_actor_cache.clear()

    def update_cache_and_solo_overlay(self, shape_cache_count, slice_cache_count, solo_label=None):
        """
        Display cache counters and solo label on the 3D view.
        shape_cache_count: number of slices in shape cache
        slice_cache_count: number of slices in pixmap cache
        solo_label: str when in solo mode, None otherwise
        """
        def _ensure_cache_actor():
            if self._cache_text_actor is None:
                self._cache_text_actor = vtk.vtkTextActor()
                self._cache_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                self._cache_text_actor.GetPositionCoordinate().SetValue(0.98, 0.95)
                tp = self._cache_text_actor.GetTextProperty()
                tp.SetColor(0.0, 0.0, 0.0)
                tp.SetFontSize(14)
                tp.SetBold(True)
                tp.SetJustification(vtk.VTK_TEXT_RIGHT)
                tp.SetVerticalJustification(vtk.VTK_TEXT_TOP)
            return self._cache_text_actor

        def _ensure_solo_actor():
            if self._solo_text_actor is None:
                self._solo_text_actor = vtk.vtkTextActor()
                self._solo_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                self._solo_text_actor.GetPositionCoordinate().SetValue(0.98, 0.92)
                tp = self._solo_text_actor.GetTextProperty()
                tp.SetColor(0.8, 0.4, 0.0)  # orange for solo
                tp.SetFontSize(16)
                tp.SetBold(True)
                tp.SetJustification(vtk.VTK_TEXT_RIGHT)
                tp.SetVerticalJustification(vtk.VTK_TEXT_TOP)
            return self._solo_text_actor

        # Cache counter
        cache_actor = _ensure_cache_actor()
        cache_actor.SetInput(f"Shapes: {shape_cache_count} | Slices: {slice_cache_count}")
        self.renderer.RemoveActor2D(cache_actor)
        self.renderer.AddActor2D(cache_actor)

        # Solo label
        solo_actor = _ensure_solo_actor()
        if solo_label:
            solo_actor.SetInput(f"Solo: {solo_label}")
            solo_actor.VisibilityOn()
            self.renderer.RemoveActor2D(solo_actor)
            self.renderer.AddActor2D(solo_actor)
        else:
            solo_actor.VisibilityOff()

        self.vtkWidget.GetRenderWindow().Render()

    def update_label_overlay(self, total_count, current_label):
        """
        Display label counter in top right corner: "Label: X / Y"
        current_label: int when viewing single label, None when viewing all
        """
        if self._label_text_actor is None:
            self._label_text_actor = vtk.vtkTextActor()
            self._label_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self._label_text_actor.GetPositionCoordinate().SetValue(0.98, 0.98)
            tp = self._label_text_actor.GetTextProperty()
            tp.SetColor(0.0, 0.0, 0.0)  # solid black
            tp.SetFontSize(16)
            tp.SetBold(True)
            tp.SetJustification(vtk.VTK_TEXT_RIGHT)
            tp.SetVerticalJustification(vtk.VTK_TEXT_TOP)
        if current_label is not None:
            text = f"Label: {current_label} / {total_count}"
        else:
            text = f"Label: All / {total_count}"
        self._label_text_actor.SetInput(text)
        self._label_text_actor.VisibilityOn()
        if total_count == 0:
            self._label_text_actor.VisibilityOff()
        # Ensure actor is in renderer (may have been removed by RemoveAllViewProps)
        self.renderer.RemoveActor2D(self._label_text_actor)  # no-op if not present
        self.renderer.AddActor2D(self._label_text_actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _create_persistent_crosshair(self):
        """Called only at initialization; create crosshair actors and add to the scene."""
        color = (1.0, 0.0, 0.0)  # Red
        radius = 2.0

        # 1. Create central sphere
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(radius)
        sphere_source.SetThetaResolution(30)
        sphere_source.SetPhiResolution(30)
        self._crosshair_sources['sphere'] = sphere_source

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(1.0)

        self.renderer.AddActor(actor)
        self.crosshair_actors.append(actor)

        # 2. Create three orthogonal lines
        for axis_name in ['x', 'y', 'z']:
            line_source = vtk.vtkLineSource()
            self._crosshair_sources[axis_name] = line_source

            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_source.GetOutputPort())

            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(color)
            line_actor.GetProperty().SetLineWidth(2.0)
            line_actor.GetProperty().SetLineStipplePattern(0xF0F0)
            line_actor.GetProperty().SetLineStippleRepeatFactor(3)

            self.renderer.AddActor(line_actor)
            self.crosshair_actors.append(line_actor)

        # Initially set all of them invisible
        for actor in self.crosshair_actors:
            actor.SetVisibility(False)


    def update_crosshair_position(self, center_point, data_shape, spacing=(1.0, 1.0, 1.0)):
        """Update crosshair position and ensure it is visible.
        
        Parameters:
            center_point: (x, y, z) position for the crosshair center (in voxel coordinates)
            data_shape: Shape of the data volume
            spacing: Voxel spacing in (x, y, z) order
        """
        if not self.crosshair_actors: # Return if not created yet
            return
        depth, height, width = data_shape
        x, y, z = center_point
        
        # Apply spacing to center point coordinates
        x_scaled = x * spacing[0]
        y_scaled = y * spacing[1]
        z_scaled = z * spacing[2]

        # Update sphere position with a fixed small radius
        self._crosshair_sources['sphere'].SetCenter(x_scaled, y_scaled, z_scaled)
        self._crosshair_sources['sphere'].SetRadius(2.0)

        # Update the positions of the three lines with spacing applied
        self._crosshair_sources['x'].SetPoint1(0, y_scaled, z_scaled)
        self._crosshair_sources['x'].SetPoint2(width * spacing[0], y_scaled, z_scaled)

        self._crosshair_sources['y'].SetPoint1(x_scaled, 0, z_scaled)
        self._crosshair_sources['y'].SetPoint2(x_scaled, height * spacing[1], z_scaled)

        self._crosshair_sources['z'].SetPoint1(x_scaled, y_scaled, 0)
        self._crosshair_sources['z'].SetPoint2(x_scaled, y_scaled, depth * spacing[2])
        
        # Keep line width fixed
        for i in range(1, 4):  # crosshair_actors[1:4] are the line actors
            self.crosshair_actors[i].GetProperty().SetLineWidth(2.0)

        # Ensure all crosshair actors are visible
        if not self.crosshair_actors[0].GetVisibility():
            for actor in self.crosshair_actors:
                actor.SetVisibility(True)

        self.vtkWidget.GetRenderWindow().Render()

    def toggle_label_visibility(self, label, visible, render=True):
        """
        Toggle the visibility of a specified label in the 3D rendered scene.

        Parameters:
            label (int): The label value to show or hide.
            visible (bool): True to show the label, False to hide it.
            render (bool): If True, render after the change. Set False for batch updates.
        """
        # Iterate over all actors in the renderer
        actors = self.renderer.GetActors()
        actors.InitTraversal()

        actor = actors.GetNextActor()
        while actor:
            # Check if the actor's label matches the specified label
            if hasattr(actor, "label") and actor.label == label:
                actor.SetVisibility(visible)  # Set visibility
            actor = actors.GetNextActor()

        # Refresh the render window to apply changes (skip if batching)
        if render:
            self.vtkWidget.GetRenderWindow().Render()
    
    def set_labels_visibility_batch(self, visibility_dict: dict):
        """
        Set visibility for multiple labels at once (single render at the end).
        
        Parameters:
            visibility_dict: Dict mapping label (int) -> visible (bool)
        """
        if not visibility_dict:
            return
            
        # Build a lookup set for faster checking
        hidden_labels = {label for label, visible in visibility_dict.items() if not visible}
        visible_labels = {label for label, visible in visibility_dict.items() if visible}
        
        # Iterate over all actors and update their visibility
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        
        actor = actors.GetNextActor()
        while actor:
            if hasattr(actor, "label"):
                label = actor.label
                if label in hidden_labels:
                    actor.SetVisibility(False)
                elif label in visible_labels:
                    actor.SetVisibility(True)
            actor = actors.GetNextActor()
        
        # Single render at the end
        self.vtkWidget.GetRenderWindow().Render()

    def add_grid(self, data: np.ndarray, spacing=(1.0, 1.0, 1.0)):
        """
        Add a coordinate grid to the 3D scene based on the input data's shape and spacing.

        Parameters:
            data (np.ndarray): 3D numpy array to determine grid bounds.
            spacing (tuple): Voxel spacing in (x, y, z) order.
        """
        # Get the bounds from the data shape and apply spacing
        depth, height, width = data.shape
        bounds = [
            0, width * spacing[0],      # x range
            0, height * spacing[1],     # y range
            0, depth * spacing[2]       # z range
        ]
        if self._axes_actor is None:
            # Create a vtkCubeAxesActor
            axes = vtk.vtkCubeAxesActor()
            axes.SetBounds(bounds)
            axes.SetCamera(self.renderer.GetActiveCamera())  # Bind to the renderer's camera

            # Set axis titles
            axes.SetXTitle("X Axis")
            axes.SetYTitle("Y Axis")
            axes.SetZTitle("Z Axis")

            # Set the deep blue color (RGB: 0.1, 0.1, 0.6)
            deep_blue = (0.1, 0.1, 0.6)

            # Set color for gridlines
            axes.GetXAxesGridlinesProperty().SetColor(*deep_blue)  # X gridlines
            axes.GetYAxesGridlinesProperty().SetColor(*deep_blue)  # Y gridlines
            axes.GetZAxesGridlinesProperty().SetColor(*deep_blue)  # Z gridlines

            # Customize gridline colors (optional)
            axes.GetXAxesLinesProperty().SetColor(1, 0, 0)  # Red for X grid
            axes.GetYAxesLinesProperty().SetColor(0, 1, 0)  # Green for Y grid
            axes.GetZAxesLinesProperty().SetColor(0, 0, 1)  # Blue for Z grid
            
            # Set the color of the axis titles (X, Y, Z titles)
            axes.GetTitleTextProperty(0).SetColor(0.2, 0.5, 0.8)  # X Axis title (light blue)
            axes.GetTitleTextProperty(1).SetColor(0.2, 0.5, 0.8)  # Y Axis title
            axes.GetTitleTextProperty(2).SetColor(0.2, 0.5, 0.8)  # Z Axis title

            # Set the color of the axis labels (numbers on X, Y, Z axes)
            axes.GetLabelTextProperty(0).SetColor(0.3, 0.7, 0.3)  # X Axis labels (greenish)
            axes.GetLabelTextProperty(1).SetColor(0.3, 0.7, 0.3)  # Y Axis labels
            axes.GetLabelTextProperty(2).SetColor(0.3, 0.7, 0.3)  # Z Axis labels
            self._axes_actor = axes
        self._axes_actor.SetBounds(bounds)
            # Add the axes actor to the renderer
        self.renderer.AddActor(self._axes_actor)

    def update_surface_with_smoothing(
        self,
        data: np.ndarray,
        smooth_iterations=20,
        spacing=(1.0, 1.0, 1.0),
        labels_to_render=None,
    ):
        """
        Extract and display the 3D surface (iso-surface) of the given data,
        with smoothing applied to the surface. Each label will have a unique color.
        
        Parameters:
            data: The 3D volume data.
            smooth_iterations: Number of smoothing iterations.
            spacing: Voxel spacing in (x, y, z) order.
        """
        print(f"Updating 3D surface with smoothing... spacing={spacing}")

        label_infos = _compute_label_bboxes_3d(data, labels_to_render)
        print(f"3D labels to render: {len(label_infos)}")

        # Clear previous actors to avoid overlaps
        self.renderer.RemoveAllViewProps()

        label_colormap = LABEL_COLORMAP  # Define your colormap
        actors = []
        futures = []

        with ThreadPoolExecutor() as executor:
            for label_info in label_infos:
                label = label_info[0]
                cache_key = (
                    label_info,
                    _label_roi_checksum(label_info, data),
                    smooth_iterations,
                    tuple(spacing),
                )
                actor = self._surface_actor_cache.get(cache_key)
                if actor is not None:
                    actors.append(actor)
                    continue
                futures.append(
                    (
                        cache_key,
                        executor.submit(
                            process_label_roi,
                            label_info,
                            data,
                            smooth_iterations,
                            label_colormap,
                            spacing,
                        ),
                    )
                )

            # Collect results as they complete
            for cache_key, future in futures:
                actor = future.result()
                if actor is not None:
                    self._surface_actor_cache[cache_key] = actor
                    actors.append(actor)

        # Step 2: Add actors to the renderer
        for actor in actors:
            self.renderer.AddActor(actor)
        
        for actor in self.crosshair_actors:
                self.renderer.AddActor(actor)
        # Step 3: Add coordinate grid to the renderer with spacing
        self.add_grid(data, spacing=spacing)

        # Step 4: Refresh the render window, preserving the camera view
        # Reset the camera only if it has never been initialized (first load)
        if not self.camera_initialized:
            self.renderer.ResetCamera()
            self.camera_initialized = True  # Mark as initialized

        # For subsequent updates, just call Render() without resetting the camera
        self.vtkWidget.GetRenderWindow().Render()


    def center_camera_on_point(self, point_3d):
        """
        Move the 3D camera's focal point to the given 3D point and translate the camera accordingly.
        
        :param point_3d: A tuple or list containing (x, y, z) coordinates.
        """
        # Get the current active camera
        camera = self.renderer.GetActiveCamera()
        if not camera:
            return

        # 1. Get the camera's current position and focal point
        old_position = np.array(camera.GetPosition())
        old_focal_point = np.array(camera.GetFocalPoint())

        # 2. The new focal point is the provided 3D point
        new_focal_point = np.array(point_3d)

        # 3. Compute the offset vector relative to the focal point
        #    This vector determines viewing angle and distance
        offset_vector = old_position - old_focal_point

        # 4. Compute the new camera position: new focal + same offset
        new_position = new_focal_point + offset_vector

        # 5. Set the camera's new focal point and position
        camera.SetFocalPoint(new_focal_point)
        camera.SetPosition(new_position)

        # 6. Re-render the window to apply changes immediately
        self.vtkWidget.GetRenderWindow().Render()

class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        config_file=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self._config_file = config_file

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        # ---------- Create main toolbar ----------
        self.main_toolbar = QtWidgets.QToolBar('Main', self)
        self.main_toolbar.setObjectName("mainToolbar")
        self.addToolBar(Qt.TopToolBarArea, self.main_toolbar)

        # Configure main toolbar style
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setFloatable(False)
        self.main_toolbar.setAllowedAreas(Qt.TopToolBarArea)
        self.main_toolbar.setIconSize(QtCore.QSize(32, 32))
        self.main_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.main_toolbar.setContentsMargins(0, 0, 0, 0)
        main_toolbar_layout = self.main_toolbar.layout()
        if main_toolbar_layout is not None:
            main_toolbar_layout.setContentsMargins(4, 0, 4, 0)
            main_toolbar_layout.setSpacing(4)
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False
        # Initialize early: resize/restore events can arrive during startup.
        self.zoomMode = self.FIT_WINDOW
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
        }
        # Kept for backward compatibility with older code paths.
        self.labelList = []

        self._noSelectionSlot = False
        self._skip_store_on_next_load = False
        self._undo_history_by_slice = collections.OrderedDict()
        self._mask_history_by_slice = collections.OrderedDict()
        self._merge_undo_stack = []  # (label1, label2, mask1) for merge undo
        self._merge_redo_stack = []  # (label1, label2, mask1) for merge redo
        self._watershed_undo_stack = []  # full tiffMask copies before 3D watershed
        self._watershed_redo_stack = []  # full tiffMask copies for watershed redo
        self._label_voxel_counts = {}
        self._labels_in_mask = set()
        self._pending_history_restore_key = None
        self._labelJumpInProgress = False
        self._last_undo_redo = None
        self._mask_autosave_dirty = False
        self._mask_edit_revision = 0
        self._last_autosave_revision = -1

        self._copied_shapes = None
        self._lastCanvasContextMenuPos = None

        # Label lifecycle state management
        self.labelMetadataStore = LabelMetadataStore()
        
        # Label visibility management
        self.visibilityManager = LabelVisibilityManager(self.labelMetadataStore)
        self.visibilityManager.allVisibilityChanged.connect(self._onAllVisibilityChanged)
        self.visibilityManager.effectiveVisibilityChanged.connect(self._onEffectiveVisibilityChanged)
        self.visibilityManager.soloModeChanged.connect(self._onSoloModeChanged)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        # Labellist for current slice of tiff data
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        
        # Connect label visibility change signal
        self.uniqLabelList.labelVisibilityChanged.connect(self.onUniqLabelVisibilityChanged)
        
        # Connect label lifecycle signals
        self.uniqLabelList.set_metadata_store(self.labelMetadataStore)
        self.uniqLabelList.set_visibility_manager(self.visibilityManager)
        self.uniqLabelList.labelVerifyRequested.connect(self.verifyLabel)
        self.uniqLabelList.labelUnverifyRequested.connect(self.unverifyLabel)
        self.uniqLabelList.labelRejectRequested.connect(self.rejectLabel)
        self.uniqLabelList.labelRevertRequested.connect(self.revertLabelToProposed)
        
        # Connect visibility quick action signals
        self.uniqLabelList.soloCurrentRequested.connect(self._onSoloCurrentRequested)
        self.uniqLabelList.showAllRequested.connect(self._onShowAllRequested)
        
        # Connect double-click to jump to label's middle slice
        self.uniqLabelList.labelDoubleClicked.connect(self._onLabelDoubleClicked)
        
        # Connect selection change to update label counter
        self.uniqLabelList.itemSelectionChanged.connect(self._updateLabelCounter)
        
        if self._config["labels"]:
            for label in self._config["labels"]:
                rgb = self._get_rgb_by_label(label)
                item = self.uniqLabelList.createItemFromLabel(label, rgb=rgb, checked=True)
                self.uniqLabelList.addItem(item)
                self.uniqLabelList.setItemLabel(item, label, rgb)

        # Create container widget with sorting controls
        label_container = QtWidgets.QWidget()
        label_layout = QtWidgets.QVBoxLayout(label_container)
        label_layout.setContentsMargins(5, 5, 5, 5)
        label_layout.setSpacing(5)
        
        # ---- Filters & Options (collapsed into dropdown) ----
        # "Show:" dropdown for list filtering
        self.listFilterCombo = QtWidgets.QComboBox()
        self.listFilterCombo.addItem("All", LabelFilterMode.ALL)
        self.listFilterCombo.addItem("Proposed", LabelFilterMode.PROPOSED)
        self.listFilterCombo.addItem("Edited", LabelFilterMode.EDITED)
        self.listFilterCombo.addItem("Verified", LabelFilterMode.VERIFIED)
        self.listFilterCombo.addItem("Not Verified", LabelFilterMode.NOT_VERIFIED)
        self.listFilterCombo.setToolTip("Filter which labels appear in the list")
        self.listFilterCombo.currentIndexChanged.connect(self._onListFilterChanged)
        
        # "Hide VERIFIED in views" checkbox
        self.hideVerifiedCheckbox = QtWidgets.QCheckBox("Hide Verified")
        self.hideVerifiedCheckbox.setToolTip("Hide VERIFIED labels in 2D/3D views (H)")
        self.hideVerifiedCheckbox.setChecked(True)  # Default ON
        self.hideVerifiedCheckbox.stateChanged.connect(self._onHideVerifiedChanged)
        
        solo_btn = QtWidgets.QPushButton("Solo")
        solo_btn.setToolTip("Show only selected label in views (S)")
        solo_btn.clicked.connect(self._onSoloCurrentFromButton)
        
        show_all_btn = QtWidgets.QPushButton("Show All")
        show_all_btn.setToolTip("Show all labels in views")
        show_all_btn.clicked.connect(self._onShowAllRequested)
        
        self.soloModeLabel = QtWidgets.QLabel("")
        self.soloModeLabel.setStyleSheet("color: orange; font-weight: bold;")
        
        sort_id_asc_btn = QtWidgets.QPushButton("↑ ID")
        sort_id_asc_btn.setToolTip("Sort by label ID (ascending)")
        sort_id_asc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_label_id(ascending=True))
        sort_id_desc_btn = QtWidgets.QPushButton("↓ ID")
        sort_id_desc_btn.setToolTip("Sort by label ID (descending)")
        sort_id_desc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_label_id(ascending=False))
        sort_size_asc_btn = QtWidgets.QPushButton("↑ Size")
        sort_size_asc_btn.setToolTip("Sort by voxel size (ascending)")
        sort_size_asc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_voxel_size(ascending=True))
        sort_size_desc_btn = QtWidgets.QPushButton("↓ Size")
        sort_size_desc_btn.setToolTip("Sort by voxel size (descending)")
        sort_size_desc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_voxel_size(ascending=False))
        sort_state_btn = QtWidgets.QPushButton("State")
        sort_state_btn.setToolTip("Sort by state (Proposed → Edited → Verified)")
        sort_state_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_state())
        
        verify_btn = QtWidgets.QPushButton("✓ Verify")
        verify_btn.setToolTip("Verify selected label (V)")
        verify_btn.clicked.connect(self.verifySelectedLabel)
        revert_btn = QtWidgets.QPushButton("⟲ Revert")
        revert_btn.setToolTip("Revert selected label to proposed state (R)")
        revert_btn.clicked.connect(self.revertSelectedLabel)
        reject_btn = QtWidgets.QPushButton("✗ Reject")
        reject_btn.setToolTip("Reject/delete selected label (Del)")
        reject_btn.clicked.connect(self.rejectSelectedLabel)
        commit_btn = QtWidgets.QPushButton("Commit")
        commit_btn.setToolTip("Commit all changes to final mask (Ctrl+Enter)")
        commit_btn.clicked.connect(self.commitChanges)
        
        self.labelStateStatsLabel = QtWidgets.QLabel("○0 ◐0 ●0")
        self.labelStateStatsLabel.setToolTip("○ Proposed  ◐ Edited  ● Verified")

        self.labelCounterLabel = QtWidgets.QLabel("— of —")
        self.labelCounterLabel.setToolTip("Current label index and total label count")
        
        # Build filters dropdown menu content
        filters_popup_widget = QtWidgets.QWidget()
        filters_popup_widget.setMinimumWidth(380)
        filters_popup_widget.setMinimumHeight(120)
        filters_popup_layout = QtWidgets.QVBoxLayout(filters_popup_widget)
        filters_popup_layout.setContentsMargins(8, 8, 8, 8)
        filters_popup_layout.setSpacing(4)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Show:"))
        filter_row.addWidget(self.listFilterCombo)
        filter_row.addWidget(self.hideVerifiedCheckbox)
        filter_row.addStretch()
        filters_popup_layout.addLayout(filter_row)
        vis_row = QtWidgets.QHBoxLayout()
        vis_row.addWidget(solo_btn)
        vis_row.addWidget(show_all_btn)
        vis_row.addWidget(self.soloModeLabel)
        vis_row.addStretch()
        filters_popup_layout.addLayout(vis_row)
        sort_row = QtWidgets.QHBoxLayout()
        sort_row.addWidget(sort_id_asc_btn)
        sort_row.addWidget(sort_id_desc_btn)
        sort_row.addWidget(sort_size_asc_btn)
        sort_row.addWidget(sort_size_desc_btn)
        sort_row.addWidget(sort_state_btn)
        sort_row.addStretch()
        filters_popup_layout.addLayout(sort_row)
        lifecycle_row = QtWidgets.QHBoxLayout()
        lifecycle_row.addWidget(verify_btn)
        lifecycle_row.addWidget(revert_btn)
        lifecycle_row.addWidget(reject_btn)
        lifecycle_row.addWidget(commit_btn)
        lifecycle_row.addStretch()
        filters_popup_layout.addLayout(lifecycle_row)
        
        filters_menu = QtWidgets.QMenu(self)
        filters_menu_action = QtWidgets.QWidgetAction(self)
        filters_menu_action.setDefaultWidget(filters_popup_widget)
        filters_menu.addAction(filters_menu_action)
        
        filters_btn = QtWidgets.QPushButton("Filters Options")
        filters_btn.setToolTip("Filter, sort, visibility, lifecycle controls")
        filters_btn.setMenu(filters_menu)
        
        # Compact top row: Filters dropdown + stats + label counter
        compact_header = QtWidgets.QHBoxLayout()
        compact_header.addWidget(filters_btn)
        compact_header.addWidget(self.labelStateStatsLabel)
        compact_header.addWidget(self.labelCounterLabel)
        compact_header.addStretch()
        
        # Search Box
        search_layout = QtWidgets.QHBoxLayout()
        self.labelSearchBox = QtWidgets.QLineEdit()
        self.labelSearchBox.setValidator(QRegExpValidator(QRegExp(r"\d*")))
        self.labelSearchBox.setPlaceholderText("Search label ID... (Enter to jump)")
        self.labelSearchBox.setToolTip("Search for a label by ID. Press Enter to jump to its middle slice.")
        self.labelSearchBox.textChanged.connect(self._onLabelSearchChanged)
        self.labelSearchBox.returnPressed.connect(self._onLabelSearchEnter)
        self.labelSearchBox.installEventFilter(self)  # Block key repeat for Enter (prevents crash when spamming)
        self.uniqLabelList.installEventFilter(self)   # Block key repeat for Enter on label list

        clear_search_btn = QtWidgets.QPushButton("✕")
        clear_search_btn.setFixedWidth(25)
        clear_search_btn.setToolTip("Clear search")
        clear_search_btn.clicked.connect(self._clearLabelSearch)
        
        search_layout.addWidget(self.labelSearchBox)
        search_layout.addWidget(clear_search_btn)
        
        label_layout.addLayout(compact_header)
        label_layout.addLayout(search_layout)
        label_layout.addWidget(self.uniqLabelList)

        # Labels dock: label list on the right side
        self.label_dock = QtWidgets.QDockWidget(self.tr("Labels"), self)
        self.label_dock.setObjectName("Labels")
        self.label_dock.setWidget(label_container)

        config_path = self._config_file
        if config_path is None or not isinstance(config_path, str) or "\n" in config_path:
            config_path = get_user_config_path()
        else:
            config_path = osp.expanduser(config_path)
        self._shortcuts_config_path = config_path
        self.shortcuts_widget = ShortcutSettingsWidget(
            self,
            shortcuts=self._config.get("shortcuts", {}),
            config_path=config_path,
            on_save=self._reloadShortcuts,
        )
        self.shortcuts_widget.shortcutsSaved.connect(self._reloadShortcuts)
        self.shortcuts_dock = QtWidgets.QDockWidget(self.tr("Keyboard Shortcuts"), self)
        self.shortcuts_dock.setObjectName("Keyboard Shortcuts")
        self.shortcuts_dock.setWidget(self.shortcuts_widget)

        self.zoomWidget = ZoomWidget()
        self._zoom_value_float = float(self.zoomWidget.value())
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.setCurrentViewAxis(0)  # Initialize canvas view axis to axial
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(
                f"Mouse is at: slice={self.currentSliceIndex}, x={round(pos.x())}, y={round(pos.y())}," 
                f" intensity={self.get_intensity_at(pos)}," 
                f" label={self.get_mask_value_at(pos)}"
            )
        )
        self.canvas.pointSelected.connect(self.pointSelectionChanged)
        self.canvas.watershedSeedClicked.connect(self.handleWatershedSeedClick)
        self.canvas.contextMenuAboutToShow.connect(self._onCanvasContextMenuAboutToShow)

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.wheelEvent = lambda event: self.wheelEvent(event)
        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.undoShapesChanged.connect(self.onUndoShapesChanged)
        self.canvas.installEventFilter(self)  # Block key repeat for Enter (prevents crash when spamming)

        # Create a horizontal splitter to arrange the 3D and image display areas side by side
        main_splitter = QSplitter(Qt.Horizontal)

        # Initialize the VTKWidget for the 3D rendering area
        self.vtk_widget = VTKSurfaceWidget(self) #VTKWidget(self)

        
        # Add the 3D rendering area and the image display area to the splitter
        main_splitter.addWidget(self.scrollArea)  # Right: Image display area
        main_splitter.addWidget(self.vtk_widget)  # Left: 3D rendering window
        # Set initial size proportions: Left takes 1, Right takes 3
        main_splitter.setStretchFactor(0, 2)  # Left widget (VTK) takes proportion 1
        main_splitter.setStretchFactor(1, 1)  # Right widget (image) takes proportion 3
        main_splitter.setSizes([800, 400])  # Initial pixel sizes for left and right widgets
        # Set the splitter as the central widget of the main window
        self.setCentralWidget(main_splitter)

        # Initialize the VTK interactor to enable user interaction
        self.vtk_widget.interactor.Initialize()

        for dock in ["label_dock", "shortcuts_dock"]:
            features = QtWidgets.QDockWidget.DockWidgetFeatures()
            dock_config = self._config.get(dock, {})
            if dock_config.get("closable", True):
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if dock_config.get("floatable", True):
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if dock_config.get("movable", True):
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if dock_config.get("show", True) is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shortcuts_dock)
        self.shortcuts_dock.setFloating(True)
        self._shortcuts_dock_was_floating = True  # Track so we re-float when reopened
        self.shortcuts_dock.visibilityChanged.connect(self._onShortcutsDockVisibilityChanged)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        
        # --- Reworked layout for label operations ---
        label_ops_widget = QWidget(self)
        main_v_layout = QVBoxLayout(label_ops_widget)
        main_v_layout.setContentsMargins(0, 0, 0, 0)
        main_v_layout.setSpacing(4)

        # Top row with two columns
        top_h_layout = QHBoxLayout()
        top_h_layout.setContentsMargins(0, 0, 0, 0)
        
        # Column 1
        col1_layout = QVBoxLayout()
        self.label_input = QLineEdit(self)
        self.label_input.setValidator(QRegExpValidator(QRegExp(r"\d*")))
        self.label_input.setPlaceholderText("Label")
        self.delete_label_button = QPushButton("Delete Label", self)
        self.delete_label_button.clicked.connect(self.delete_label)
        self.split_label_button = QPushButton("Split Label", self)
        self.split_label_button.clicked.connect(self.split_label)
        col1_layout.addWidget(self.label_input)
        col1_layout.addWidget(self.delete_label_button)
        col1_layout.addWidget(self.split_label_button)
        top_h_layout.addLayout(col1_layout)

        # 3D Watershed UI controls
        watershed_3d_layout = QHBoxLayout()
        self.watershed_3d_label_input = QLineEdit(self)
        self.watershed_3d_label_input.setPlaceholderText("Auto-detected from seeds")
        self.watershed_3d_label_input.setReadOnly(True)  # Set read-only
        self.watershed_3d_clear_button = QPushButton("Clear Seeds", self)
        self.watershed_3d_apply_button = QPushButton("Apply 3D Watershed", self)
        
        self.watershed_3d_clear_button.clicked.connect(self.clear_watershed_seeds)
        self.watershed_3d_apply_button.clicked.connect(self.apply_3d_watershed)
        
        watershed_3d_layout.addWidget(QLabel("Watershed Label:"))
        watershed_3d_layout.addWidget(self.watershed_3d_label_input)
        watershed_3d_layout.addWidget(self.watershed_3d_clear_button)
        watershed_3d_layout.addWidget(self.watershed_3d_apply_button)

        main_v_layout.addLayout(top_h_layout)
        main_v_layout.addLayout(watershed_3d_layout)  # Add 3D watershed controls


        label_ops_action = QWidgetAction(self)
        label_ops_action.setDefaultWidget(label_ops_widget)

        # --- Begin vertical Merge Label widget ---
        merge_label_widget = QWidget(self)
        # no fixed width: let it size to contents

        v_layout = QVBoxLayout(merge_label_widget)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(2)
        v_layout.setAlignment(Qt.AlignLeft)

        # Label transparency slider (above merge labels)
        transparency_row = QHBoxLayout()
        transparency_row.setContentsMargins(0, 0, 0, 0)
        transparency_row.setSpacing(4)
        transparency_label = QLabel("Label opacity:")
        transparency_label.setFixedHeight(16)
        transparency_row.addWidget(transparency_label)
        self.label_opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.label_opacity_slider.setRange(0, 100)
        self.label_opacity_slider.setValue(100)
        self.label_opacity_slider.setFixedHeight(18)
        self.label_opacity_slider.setFixedWidth(80)
        self.label_opacity_slider.valueChanged.connect(self._on_label_opacity_changed)
        transparency_row.addWidget(self.label_opacity_slider)
        v_layout.addLayout(transparency_row)

        # Row 1: two inputs + arrow
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)

        self.merge_label_input_1 = QLineEdit(self)
        self.merge_label_input_1.setValidator(QRegExpValidator(QRegExp(r"\d*")))
        self.merge_label_input_1.setPlaceholderText("L1")
        self.merge_label_input_1.setFixedWidth(30)
        input_layout.addWidget(self.merge_label_input_1)

        arrow_label = QLabel("→")
        arrow_label.setContentsMargins(0, 0, 0, 0)
        # size arrow to its glyph width
        w = arrow_label.fontMetrics().horizontalAdvance("→")
        arrow_label.setFixedWidth(w)
        input_layout.addWidget(arrow_label)

        self.merge_label_input_2 = QLineEdit(self)
        self.merge_label_input_2.setValidator(QRegExpValidator(QRegExp(r"\d*")))
        self.merge_label_input_2.setPlaceholderText("L2")
        self.merge_label_input_2.setFixedWidth(30)
        input_layout.addWidget(self.merge_label_input_2)

        v_layout.addLayout(input_layout)

        # Row 2: Merge button, left‐aligned
        self.merge_label_button = QPushButton("Merge Labels", self)
        self.merge_label_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.merge_label_button.clicked.connect(self.merge_labels)
        v_layout.addWidget(self.merge_label_button, alignment=Qt.AlignLeft)

        merge_labels_action = QWidgetAction(self)
        merge_labels_action.setDefaultWidget(merge_label_widget)
        # --- End merge widget ---

        # Create brush controls
        # Create a brush widget and set up a vertical layout
        brush_widget = QtWidgets.QWidget()
        brush_widget.setFixedWidth(120)  # Set the total width

        # Use a vertical layout
        brush_layout = QtWidgets.QVBoxLayout()
        brush_layout.setContentsMargins(2, 0, 2, 0)  # Minimize margins
        brush_layout.setSpacing(2)  # Set element spacing to 2px

        # Brush Size label (centered at the top)
        brush_size_label = QtWidgets.QLabel("Brush Size")
        brush_size_label.setAlignment(Qt.AlignCenter)  # Center the text
        brush_size_label.setFixedHeight(15)  # Set a fixed label height
        brush_layout.addWidget(brush_size_label)

        # Brush size slider (placed below the label)
        self.brush_size_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(
            lambda v: self.canvas.setBrushSize(v)
        )
        self.brush_size_slider.setFixedHeight(20)  # Set the slider height
        brush_layout.addWidget(self.brush_size_slider)

        # Add an input field for the label using QLineEdit
        self.brush_label_input = QtWidgets.QLineEdit()
        self.brush_label_input.setValidator(QRegExpValidator(QRegExp(r"\d*")))
        self.brush_label_input.setPlaceholderText("Enter label")
        self.brush_label_input.setFixedHeight(20)  # Set the input field height
        brush_layout.addWidget(self.brush_label_input)

        # Set the layout for the brush widget
        brush_widget.setLayout(brush_layout)

        # Compact style settings
        brush_widget.setStyleSheet("""
            QSlider {
                margin: 0;
                padding: 0;
            }
            QLabel {
                font-size: 9px;  /* Reduce font size */
                margin: 0;
                padding: 0;
            }
            QLineEdit {
                font-size: 9px;
                margin: 0;
                padding: 0;
            }
        """)

        brush_action = QtWidgets.QWidgetAction(self)
        brush_action.setDefaultWidget(brush_widget)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        

        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        openPrevImg = action(
            self.tr("&Prev Slice"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open previous slice (hold Ctl+Shift to copy labels)"),
            enabled=True,
        )
        openNextImg = action(
            self.tr("&Next Slice"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next slice (hold Ctl+Shift to copy labels)"),
            enabled=True,
        )
        openPrevTenImg = action(
            self.tr("&Prev 10"),
            self.openPrevTenImg,
            None,  # No shortcut for Prev 10
            "prev",
            self.tr("Open prev 10 slices (hold Ctl+Shift to copy labels)"),
            enabled=True,
        )
        saveMask = action(
            self.tr("&Save Mask"),
            self.saveMask,
            shortcuts["save"],
            "save",
            self.tr("Save mask to  tiff file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: self._setCanvasAiModelForMode("ai_polygon")
        )
        createAiMaskMode = action(
            self.tr("Points AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            shortcuts.get("create_ai_mask_mode"),
            "objects",
            self.tr("Start drawing ai_mask by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self._setCanvasAiModelForMode("ai_mask")
        )
        createBoxAiMaskMode = action(
            self.tr("Box AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts.get("create_rectangle_mode"),
            "objects",
            self.tr("Draw a box and use it as the AI mask prompt."),
            enabled=False,
        )
        createBoxAiMaskMode.changed.connect(
            lambda: self._setCanvasAiModelForMode("rectangle")
        )
        createAiBoundaryMode = action(
            self.tr("AI Boundary"),
            lambda: self.toggleDrawMode(False, createMode="ai_boundary"),
            shortcuts.get("create_ai_boundary_mode"),
            "objects",
            self.tr("Start drawing ai_boundary by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiBoundaryMode.changed.connect(
            lambda: self._setCanvasAiModelForMode("ai_boundary")
        )

        # Add brush mode action
        createBrushMode = action(
            self.tr("Brush Mode"),
            lambda: self.toggleDrawMode(False, createMode="brush"),
            shortcuts.get("create_brush_mode"),
            "objects",
            self.tr("Start freehand drawing with brush"),
            enabled=False,
        )
        createBoxEraseMode = action(
            self.tr("Box Erase"),
            lambda: self.toggleDrawMode(False, createMode="erase"),
            shortcuts.get("erase_mode"),
            "objects",
            self.tr("Draw a box to erase labels in that region."),
            enabled=False,
        )
        createWatershed3dMode = action(
            self.tr("Watershed Seeds"),
            lambda: self.toggleDrawMode(False, createMode="watershed_3d"),
            shortcuts.get("create_watershed_3d_mode"),
            "objects",
            self.tr("Click to place seed points for 3D watershed"),
            enabled=False,
        )
        verifyLabelAtCursorAction = action(
            self.tr("✓ Verify label at cursor"),
            self.verifyLabelAtCursor,
            None,
            None,
            self.tr("Verify the label under the cursor (right-click position)"),
            enabled=True,
        )
        unverifyLabelAtCursorAction = action(
            self.tr("↩ Unverify label at cursor"),
            self.unverifyLabelAtCursor,
            None,
            None,
            self.tr("Unverify the label under the cursor (right-click position)"),
            enabled=True,
        )
        soloLabelAtCursorAction = action(
            self.tr("👁 Solo label at cursor"),
            self.soloLabelAtCursor,
            None,
            None,
            self.tr("Show only the label under the cursor (right-click position)"),
            enabled=True,
        )
        selectMode = action(
            self.tr("View /Select"),
            lambda: self.toggleDrawMode(edit=True),  # Call toggleDrawMode(True) to exit drawing
            shortcuts.get("select_mode", "V"),
            "objects",  # Use an icon representing "select"
            self.tr("Exit drawing and enter selection mode"),
            enabled=True,
            checkable=True,  # Set as checkable
        )
        # Create an action group to manage all mode buttons
        self.mode_action_group = QtWidgets.QActionGroup(self)
        self.mode_action_group.setExclusive(True)  # Exclusive so only one is selected
        self.mode_action_group.addAction(selectMode)
        self.mode_action_group.addAction(createAiMaskMode)
        self.mode_action_group.addAction(createBoxAiMaskMode)
        self.mode_action_group.addAction(createAiBoundaryMode)
        self.mode_action_group.addAction(createBrushMode)
        self.mode_action_group.addAction(createBoxEraseMode)
        self.mode_action_group.addAction(createWatershed3dMode)

        # Store this new action in self.actions

        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        
        # Undo/Redo actions for shape operations
        undo = action(
            self.tr("Undo"),
            self.undoEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last shape edit"),
            enabled=True,
        )
        
        redo = action(
            self.tr("Redo"),
            self.redoEdit,
            shortcuts.get("redo", "Ctrl+Shift+Z"),
            "undo",
            self.tr("Redo last undone shape edit"),
            enabled=True,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )


        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)


        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Scrollable drawing tools (View/Select, Point AI Mask, Box AI-Mask, etc.)
        drawing_tools_container = QWidget(self)
        drawing_tools_layout = QHBoxLayout(drawing_tools_container)
        drawing_tools_layout.setContentsMargins(0, 0, 0, 0)
        drawing_tools_layout.setSpacing(2)
        for act in (
            selectMode,
            createAiMaskMode,
            createBoxAiMaskMode,
            createAiBoundaryMode,
            createBrushMode,
            createBoxEraseMode,
            createWatershed3dMode,
        ):
            btn = QtWidgets.QToolButton()
            btn.setDefaultAction(act)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            drawing_tools_layout.addWidget(btn)
        drawing_tools_scroll = QtWidgets.QScrollArea()
        drawing_tools_scroll.setWidget(drawing_tools_container)
        drawing_tools_scroll.setWidgetResizable(True)
        drawing_tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        drawing_tools_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        drawing_tools_scroll.setMaximumHeight(70)
        drawing_tools_scroll.setMinimumWidth(120)
        drawing_tools_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        drawing_tools_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        drawing_tools_action = QWidgetAction(self)
        drawing_tools_action.setDefaultWidget(drawing_tools_scroll)

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            saveMask=saveMask,
            open=open_,
            close=close,
            quit=quit,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            openPrevImg=openPrevImg,
            openNextImg=openNextImg,
            undoLastPoint=undoLastPoint,
            undo=undo,
            redo=redo,
            selectMode=selectMode, 
            createPointMode=createPointMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            createBoxAiMaskMode=createBoxAiMaskMode,
            createAiBoundaryMode=createAiBoundaryMode,
            createBrushMode=createBrushMode,
            createBoxEraseMode=createBoxEraseMode,
            createWatershed3dMode=createWatershed3dMode,
            zoom=zoom,
            fileMenuActions=(open_, close, quit),
            tool=(
                drawing_tools_action,
                brush_action,
                label_ops_action,
                merge_labels_action,
            ),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                undo,
                redo,
                None,
                undoLastPoint,
                None,
                None,
                toggle_keep_prev_mode,
                None,
                openPrevImg,
                openNextImg,
            ),
            # menu shown at right click on canvas/slices
            menu=(
                selectMode,
                createAiMaskMode,
                createBoxAiMaskMode,
                createAiBoundaryMode,
                createBrushMode,
                createBoxEraseMode,
                createWatershed3dMode,
                None,
                verifyLabelAtCursorAction,
                unverifyLabelAtCursorAction,
                soloLabelAtCursorAction,
            ),
            onLoadActive=(
                close,
                createPointMode,
                createAiPolygonMode,
                createAiMaskMode,
                createBoxAiMaskMode,
                createBoxEraseMode,
            ),
        )


        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                self.menus.recentFiles,
                None,
                openPrevImg,
                openNextImg,
                None,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.label_dock.toggleViewAction(),
                self.shortcuts_dock.toggleViewAction(),
                None,
                fill_drawing,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget (solo feature: Show All on right-click):
        self._showAllCanvasAction = action(
            "👁 Show All (Solo All)",
            self._onShowAllRequested,
            tip="Show all labels and exit solo mode",
        )
        utils.addActions(
            self.canvas.menus[0],
            tuple(self.actions.menu) + (None, self._showAllCanvasAction),
        )
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Move here", self.moveShape),
                None,
                self._showAllCanvasAction,
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        #
        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Mask Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)
        #
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)
        model_names = [
            model.name
            for model in MODELS
            if hasattr(model, "predict_mask_from_points")
        ]
        self._selectAiModelComboBox.addItems(model_names)
        if self._config["ai"]["default"] in model_names:
            model_index = model_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: self.canvas.set_ai_model(
                self._get_or_create_ai_model(self._selectAiModelComboBox.currentText()),
                self.embedding_dir
            )
        )

        # View/3D controls for toolbar (Show All 3D, Spacing, Update 3D, Axis)
        self.showAll3D = False
        self.crosshair_center_xy = None
        view_controls_widget = QtWidgets.QWidget()
        view_controls_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        view_controls_layout = QtWidgets.QVBoxLayout(view_controls_widget)
        view_controls_layout.setContentsMargins(6, 2, 6, 2)
        view_controls_layout.setSpacing(2)
        self.checkBox3DRendering = QtWidgets.QCheckBox(self.tr("Show All 3D"))
        self.checkBox3DRendering.setChecked(self.showAll3D)
        self.checkBox3DRendering.stateChanged.connect(self.on3DRenderingCheckBoxChanged)
        self.checkBox3DRendering.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        view_controls_layout.addWidget(self.checkBox3DRendering)
        spacing_layout = QtWidgets.QHBoxLayout()
        spacing_layout.setContentsMargins(0, 0, 0, 0)
        spacing_layout.setSpacing(4)
        spacing_layout.addWidget(QtWidgets.QLabel(self.tr("Spacing:")))
        self.spacing_x_input = QtWidgets.QLineEdit()
        self.spacing_x_input.setValidator(QRegExpValidator(QRegExp(r"\d*\.?\d*")))
        self.spacing_x_input.setText("1")
        self.spacing_x_input.setMaximumWidth(40)
        self.spacing_x_input.setPlaceholderText("X")
        spacing_layout.addWidget(self.spacing_x_input)
        self.spacing_y_input = QtWidgets.QLineEdit()
        self.spacing_y_input.setValidator(QRegExpValidator(QRegExp(r"\d*\.?\d*")))
        self.spacing_y_input.setText("1")
        self.spacing_y_input.setMaximumWidth(40)
        self.spacing_y_input.setPlaceholderText("Y")
        spacing_layout.addWidget(self.spacing_y_input)
        self.spacing_z_input = QtWidgets.QLineEdit()
        self.spacing_z_input.setValidator(QRegExpValidator(QRegExp(r"\d*\.?\d*")))
        self.spacing_z_input.setText("1")
        self.spacing_z_input.setMaximumWidth(40)
        self.spacing_z_input.setPlaceholderText("Z")
        spacing_layout.addWidget(self.spacing_z_input)
        view_controls_layout.addLayout(spacing_layout)
        self.update3DButton = QtWidgets.QPushButton(self.tr("Update 3D"))
        self.update3DButton.clicked.connect(self.update3D)
        self.update3DButton.setFixedHeight(26)
        self.update3DButton.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        view_controls_layout.addWidget(self.update3DButton)
        self.viewSelection = QtWidgets.QComboBox()
        self.viewSelection.addItems(["Axial", "Coronal", "Sagittal"])
        self.viewSelection.currentIndexChanged.connect(self.updateViewAxis)
        self.viewSelection.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.viewSelection.setMinimumContentsLength(0)
        self.viewSelection.setMaximumWidth(120)
        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addWidget(QtWidgets.QLabel(self.tr("Axis:")))
        axis_layout.addWidget(self.viewSelection)
        view_controls_layout.addLayout(axis_layout)
        view_3d_controls_action = QtWidgets.QWidgetAction(self)
        view_3d_controls_action.setDefaultWidget(view_controls_widget)

        # Create the segmentation model widget for the segmentation dock
        segmentallWidget = QtWidgets.QWidget()

        # Use QVBoxLayout for the overall layout
        mainLayout = QtWidgets.QVBoxLayout(segmentallWidget)
        mainLayout.setContentsMargins(2, 2, 2, 2)
        mainLayout.setSpacing(2)

        # Add label for the model selector (First row)
        segmentallLabel = QtWidgets.QLabel(self.tr("Segmentation Model"))
        segmentallLabel.setAlignment(QtCore.Qt.AlignCenter)
        mainLayout.addWidget(segmentallLabel)

        # Add model selection dropdown (Second row)
        self._segmentallComboBox = QtWidgets.QComboBox()
        mainLayout.addWidget(self._segmentallComboBox)

        # Populate the dropdown with available segment-all models only.
        model_options = [
            model.name
            for model in MODELS
            if hasattr(model, "predict") and getattr(model, "available", True)
        ]
        self._segment_all_available = bool(model_options)
        if not model_options:
            model_options = ["Unavailable"]
        self._segmentallComboBox.addItems(model_options)

        # Set the default model
        default_model = self._config["segment_all"]["default"]
        default_model_lower = default_model.lower()
        normalized_options = [name.lower() for name in model_options]
        if default_model_lower in normalized_options:
            model_index = normalized_options.index(default_model_lower)
        else:
            logger.warning(
                "Default segmentation model is not found: %r",
                default_model,
            )
            model_index = 0
        self._segmentallComboBox.setCurrentIndex(model_index)

        # Add buttons (Third row)
        buttonLayout = QtWidgets.QHBoxLayout()  # Horizontal layout for buttons
        buttonLayout.setSpacing(5)  # Reduce spacing between buttons
        self.segmentAllButton = QtWidgets.QPushButton(self.tr("Segment All"))
        self.trackingButton = QtWidgets.QPushButton(self.tr("Tracking"))
        self.interpolateButton = QtWidgets.QPushButton(self.tr("Interpolate"))
        buttonLayout.addWidget(self.segmentAllButton)
        buttonLayout.addWidget(self.trackingButton)
        buttonLayout.addWidget(self.interpolateButton)
        mainLayout.addLayout(buttonLayout)

        # Connect buttons to their respective actions
        self.segmentAllButton.clicked.connect(self.segmentAll)
        if not self._segment_all_available:
            self.segmentAllButton.setEnabled(False)
            self._segmentallComboBox.setEnabled(False)
        self.trackingButton.clicked.connect(self.tracking)
        self.interpolateButton.clicked.connect(self.show_interpolate_dialog)

        # Segmentation dock: openable from View menu (like Labels and Shortcuts)
        self.segmentation_dock = QtWidgets.QDockWidget(self.tr("Segmentation"), self)
        self.segmentation_dock.setObjectName("Segmentation")
        self.segmentation_dock.setWidget(segmentallWidget)
        seg_dock_config = self._config.get("segmentation_dock", {})
        seg_features = QtWidgets.QDockWidget.DockWidgetFeatures()
        if seg_dock_config.get("closable", True):
            seg_features = seg_features | QtWidgets.QDockWidget.DockWidgetClosable
        if seg_dock_config.get("floatable", True):
            seg_features = seg_features | QtWidgets.QDockWidget.DockWidgetFloatable
        if seg_dock_config.get("movable", True):
            seg_features = seg_features | QtWidgets.QDockWidget.DockWidgetMovable
        self.segmentation_dock.setFeatures(seg_features)
        if seg_dock_config.get("show", True) is False:
            self.segmentation_dock.setVisible(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.segmentation_dock)
        # Insert segmentation dock next to keyboard shortcuts (before the separator)
        sep = self.menus.view.actions()[2]
        self.menus.view.insertAction(sep, self.segmentation_dock.toggleViewAction())

        # ---------- Add actions to main toolbar ----------
        # File / Navigation actions
        utils.addActions(self.main_toolbar, (open_, saveMask))
        self.main_toolbar.addSeparator()
        # Draw / Labels actions populated by populateModeActions() (tools)
        self.main_toolbar.addSeparator()
        utils.addActions(
            self.main_toolbar,
            (
                view_3d_controls_action,
                None,
                selectAiModel,
            ),
        )
        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.currentSliceIndex = -1
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.currentSliceIndex = -1
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value(
            "window/size",
            QtCore.QSize(1200, 900),
            type=QtCore.QSize,
        )
        position = self.settings.value(
            "window/position",
            QtCore.QPoint(100, 100),
            type=QtCore.QPoint,
        )
        state = self.settings.value(
            "window/state",
            QtCore.QByteArray(),
            type=QtCore.QByteArray,
        )
        if not isinstance(size, QtCore.QSize) or size.width() < 320 or size.height() < 240:
            size = QtCore.QSize(1200, 900)
        if not isinstance(position, QtCore.QPoint):
            position = QtCore.QPoint(100, 100)
        if state is None:
            state = QtCore.QByteArray()
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        if not state.isEmpty():
            self.restoreState(state)
        self._ensureWindowVisible()

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # Initialize cache and threading
        self.sliceCache = collections.OrderedDict()  # LRU cache for slice pixmaps (bounded)
        self.shapeCache = collections.OrderedDict()  # LRU cache for mask shapes per slice
        self.cacheThread = None  # Thread for background caching
        self.cacheRange = 5  # Number of slices to cache before and after the current slice
        self._slice_scroll_accumulator = 0  # Accumulated wheel delta for throttled scroll
        self._slice_scroll_throttle_timer = QtCore.QTimer(self)
        self._slice_scroll_throttle_timer.setSingleShot(True)
        self._slice_scroll_throttle_timer.timeout.connect(self._applyScrollAccumulator)
        self.currentSliceIndex = 0  # Current slice index
        self.currentSliceIndex = 0  # Default slice index
        self.currentViewAxis = 0  # Default axis: 0 = Axial, 1 = Coronal, 2 = Sagittal

        # initialize lastClickedPoint so it always exists
        self.lastClickedPoint = None
        # Track the last rendered 3D label to avoid re-rendering the same label
        self.lastRendered3DLabel = None
        # Track if tool has been switched to an editing tool since last 3D render
        self.toolSwitchedSince3DRender = False


        # Now that all toolbar actions are added, populate and rebuild the main toolbar
        self.populateModeActions()
        self.label_visibility_states = {}
        self.compute_thread = None
        self.compute_thread_stop_event = None 
        self.embedding_task_queue = None  
        self.ai_model_cache = {}  # Cache for AI model 
        self.recent_label = "10000"  # Store the most recent label for AI operations
        self._sliceLoadTimer = QtCore.QTimer(self)
        self._sliceLoadTimer.setSingleShot(True)
        self._sliceLoadTimer.timeout.connect(self.loadAnnotationsAndMasks)
        self._sliceLoadDelayMs = 120  # base delay
        self._sliceLoadDelayMsRapid = 200  # Tip 7: longer delay during rapid scroll
        self._handling_visibility = False
        self._maskAutosaveTimer = QtCore.QTimer(self)
        self._maskAutosaveTimer.timeout.connect(self._autosaveTempMask)
        self._maskAutosaveTimer.start(MASK_AUTOSAVE_INTERVAL_MS)
        
        # Install keyboard shortcuts
        self._installShortcuts()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions



    def populateModeActions(self):
        # 1) Find the position to insert drawing actions in main_toolbar
        # Drawing actions should be inserted after the first separator (after file/nav actions)
        toolbar_actions = self.main_toolbar.actions()
        insert_pos = 0
        separator_count = 0
        for i, act in enumerate(toolbar_actions):
            if act.isSeparator():
                separator_count += 1
                if separator_count == 1:
                    # Insert right after the first separator
                    insert_pos = i + 1
                    break
        
        # 2) Clear existing drawing actions from main_toolbar (if any)
        # Find and remove all actions between first and second separator
        to_remove = []
        in_draw_section = False
        separator_count = 0
        for act in toolbar_actions:
            if act.isSeparator():
                separator_count += 1
                if separator_count == 1:
                    in_draw_section = True
                elif separator_count == 2:
                    in_draw_section = False
            elif in_draw_section:
                to_remove.append(act)
        
        for act in to_remove:
            self.main_toolbar.removeAction(act)
        
        # 3) Insert drawing/label-related tool buttons into main_toolbar
        for i, act in enumerate(self.actions.tool):
            self.main_toolbar.insertAction(
                toolbar_actions[insert_pos] if insert_pos < len(toolbar_actions) else None,
                act
            )

        # 4) Update the Canvas context menu
        self.canvas.menus[0].clear()
        utils.addActions(
            self.canvas.menus[0],
            tuple(self.actions.menu) + (None, self._showAllCanvasAction),
        )

        # 5) Update the main window's Edit menu
        self.menus.edit.clear()
        edit_actions = (
            self.actions.createPointMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.createBoxAiMaskMode,
            self.actions.createAiBoundaryMode,
            self.actions.createBoxEraseMode,
        )
        utils.addActions(self.menus.edit, edit_actions + self.actions.editMenu)

    def setDirty(self):
        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            return
        self.dirty = True
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.createPointMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        self.actions.createBoxAiMaskMode.setEnabled(True)
        self.actions.createAiBoundaryMode.setEnabled(True)
        self.actions.createBrushMode.setEnabled(True)
        self.actions.createBoxEraseMode.setEnabled(True)
        self.actions.createWatershed3dMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)
        
        self.actions.selectMode.setChecked(True)

    def undoEdit(self):
        """Execute undo: single-slice edits first, then merge, then watershed. Each Ctrl+Z undoes one operation only."""
        self._last_undo_redo = "undo"
        # Single-slice (canvas/shape) edits first; volume operations last (each undo reverts one action)
        if self.canvas.undo():
            self.status(self.tr("Undo successful"))
            return
        if self._perform_merge_undo():
            self.status(self.tr("Undo successful"))
            return
        if self._perform_watershed_undo():
            self.status(self.tr("Undo successful"))
            return
        self._last_undo_redo = None
        self.status(self.tr("Nothing to undo"))

    def redoEdit(self):
        """Execute redo: single-slice first, then merge, then watershed. Mirrors undo order."""
        self._last_undo_redo = "redo"
        if self.canvas.redo():
            self.status(self.tr("Redo successful"))
            return
        if self._perform_merge_redo():
            self.status(self.tr("Redo successful"))
            return
        if self._perform_watershed_redo():
            self.status(self.tr("Redo successful"))
            return
        self._last_undo_redo = None
        self.status(self.tr("Nothing to redo"))

    def _perform_watershed_undo(self):
        """Undo the last 3D watershed operation. Returns True if successful."""
        if (
            not self._watershed_undo_stack
            or not hasattr(self, "tiffMask")
            or self.tiffMask is None
        ):
            return False
        prev_entry = self._watershed_undo_stack.pop()
        if isinstance(prev_entry, tuple) and prev_entry[0] == "region":
            _, bbox, prev_region = prev_entry
            z1, z2, y1, y2, x1, x2 = bbox
            current_region = self.tiffMask[z1:z2, y1:y2, x1:x2].copy()
            self._watershed_redo_stack.append(("region", bbox, current_region))
            self.tiffMask[z1:z2, y1:y2, x1:x2] = prev_region
            self._invalidate_shape_cache_for_bbox(bbox)
        else:
            prev_mask = prev_entry
            self._watershed_redo_stack.append(self.tiffMask.copy())
            self.tiffMask[...] = prev_mask
            self._invalidate_shape_cache()
        if len(self._watershed_redo_stack) > WATERSHED_UNDO_LIMIT:
            self._watershed_redo_stack.pop(0)
        self.updateUniqueLabelListFromEntireMask()
        self.loadAnnotationsAndMasks()
        self.openNextImg(nextN=0, store_history=False)
        self.setDirty()
        self._markMaskDirty()
        return True

    def _perform_watershed_redo(self):
        """Redo the last undone 3D watershed operation. Returns True if successful."""
        if (
            not self._watershed_redo_stack
            or not hasattr(self, "tiffMask")
            or self.tiffMask is None
        ):
            return False
        next_entry = self._watershed_redo_stack.pop()
        if isinstance(next_entry, tuple) and next_entry[0] == "region":
            _, bbox, next_region = next_entry
            z1, z2, y1, y2, x1, x2 = bbox
            current_region = self.tiffMask[z1:z2, y1:y2, x1:x2].copy()
            self._watershed_undo_stack.append(("region", bbox, current_region))
            self.tiffMask[z1:z2, y1:y2, x1:x2] = next_region
            self._invalidate_shape_cache_for_bbox(bbox)
        else:
            next_mask = next_entry
            self._watershed_undo_stack.append(self.tiffMask.copy())
            self.tiffMask[...] = next_mask
            self._invalidate_shape_cache()
        if len(self._watershed_undo_stack) > WATERSHED_UNDO_LIMIT:
            self._watershed_undo_stack.pop(0)
        self.updateUniqueLabelListFromEntireMask()
        self.loadAnnotationsAndMasks()
        self.openNextImg(nextN=0, store_history=False)
        self.setDirty()
        self._markMaskDirty()
        return True

    def _perform_merge_undo(self):
        """Undo the last merge operation. Returns True if successful."""
        if not self._merge_undo_stack or not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return False
        label1, label2, mask1 = self._merge_undo_stack.pop()
        source_count = int(np.count_nonzero(mask1))
        self.tiffMask[mask1] = label1
        self._invalidate_shape_cache_for_mask(mask1)
        self.labelMetadataStore.undo()
        self._merge_redo_stack.append((label1, label2, mask1))
        self._updateCachedCountsForMerge(label2, label1, source_count)
        self._updateLabelStateStats()
        self.openNextImg(nextN=0, store_history=False)
        self.setDirty()
        self._markMaskDirty()
        return True

    def _perform_merge_redo(self):
        """Redo the last undone merge operation. Returns True if successful."""
        if not self._merge_redo_stack or not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return False
        label1, label2, mask1 = self._merge_redo_stack.pop()
        source_count = int(np.count_nonzero(mask1))
        self.tiffMask[mask1] = label2
        self._invalidate_shape_cache_for_mask(mask1)
        self.labelMetadataStore.redo()
        self._merge_undo_stack.append((label1, label2, mask1))
        self._updateCachedCountsForMerge(label1, label2, source_count)
        self._updateLabelStateStats()
        self.openNextImg(nextN=0, store_history=False)
        self.setDirty()
        self._markMaskDirty()
        return True

    def onUndoShapesChanged(self):
        """
        当 canvas 执行 undo/redo 后，同步更新 label list 和 tiffMask。
        """
        # 如果有 tiffMask，优先使用 mask 快照恢复；否则再根据 shapes 重建。
        if hasattr(self, 'tiffMask') and self.tiffMask is not None:
            handled = False
            key = self._slice_key()
            history = self._mask_history_by_slice.get(key)
            if history and self._last_undo_redo == "undo" and history["undo"]:
                popped = history["undo"].pop()
                redo_entry = self._capture_mask_state_for_redo(popped)
                if redo_entry:
                    history["redo"].append(redo_entry)
                    if len(history["redo"]) > self.canvas._undo_limit:
                        history["redo"].pop(0)
                self._apply_mask_entry(popped)
                handled = True
            elif history and self._last_undo_redo == "redo" and history["redo"]:
                popped = history["redo"].pop()
                undo_entry = self._capture_mask_state_for_redo(popped)
                if undo_entry:
                    history["undo"].append(undo_entry)
                    if len(history["undo"]) > self.canvas._undo_limit:
                        history["undo"].pop(0)
                self._apply_mask_entry(popped)
                handled = True

            if not handled:
                self._rebuildCurrentSliceMask()
            else:
                self.openNextImg(nextN=0, immediate_load=True, store_history=False)

            self.updateUniqueLabelListFromEntireMask()

        self._last_undo_redo = None
        self.setDirty()
        if hasattr(self, "tiffMask") and self.tiffMask is not None:
            self._markMaskDirty()

    def _rebuildCurrentSliceMask(self):
        """
        根据当前 canvas.shapes 重建当前 slice 的 tiffMask。
        这是 undo/redo 后同步 mask 数据的关键方法。
        """
        if self.tiffMask is None:
            return
        
        slice_id = self.currentSliceIndex
        
        # 获取当前 slice 对应的 mask 切片
        if self.currentViewAxis == 0:  # Axial
            current_mask = self.tiffMask[slice_id, :, :]
        elif self.currentViewAxis == 1:  # Coronal
            current_mask = self.tiffMask[:, slice_id, :]
        else:  # Sagittal
            current_mask = self.tiffMask[:, :, slice_id]
        
        # 清空当前 slice 的 mask，再根据 shapes 重建
        current_mask[:] = 0
        
        # 重新写入当前 slice 的 shapes 到 mask
        for shape in self.canvas.shapes:
            if hasattr(shape, 'slice_id') and shape.slice_id == slice_id:
                if shape.shape_type == "mask" and shape.mask is not None:
                    try:
                        label_val = int(shape.label)
                    except (ValueError, TypeError):
                        continue
                    if len(shape.points) < 2:
                        continue
                    x1, y1 = shape.points[0].x(), shape.points[0].y()
                    x2, y2 = shape.points[1].x(), shape.points[1].y()
                    index_tuple = self.get_mask_update_index(shape.slice_id, y1, y2, x1, x2)
                    self.tiffMask[index_tuple][shape.mask > 0] = label_val
        
        # 刷新显示（不记录到撤销栈）
        self._skip_store_on_next_load = True
        self.openNextImg(nextN=0, immediate_load=True, store_history=False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def _slice_key(self, slice_index=None):
        if slice_index is None:
            slice_index = self.currentSliceIndex
        return (self.currentViewAxis, slice_index)

    def _current_axis_slice_count(self):
        if not hasattr(self, "tiffData") or self.tiffData is None:
            return 0
        return self.tiffData.shape[self.currentViewAxis]

    def _current_slice_status_text(self):
        total = self._current_axis_slice_count()
        axis_name = self.viewSelection.currentText() if hasattr(self, "viewSelection") else "Axis"
        return f"Loaded {axis_name} slice {self.currentSliceIndex}/{total}"

    def _invalidate_shape_cache_for_slice(self, slice_index=None):
        """Invalidate mask shape cache for a slice (call when tiffMask is modified)."""
        if not hasattr(self, "shapeCache"):
            return
        key = self._slice_key(slice_index)
        self.shapeCache.pop(key, None)
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            self.vtk_widget.clear_surface_cache()
        self._update_3d_cache_overlay()

    def _invalidate_shape_cache(self):
        """Invalidate entire mask shape cache (call on watershed, load new file, reset)."""
        if hasattr(self, "shapeCache"):
            self.shapeCache.clear()
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            self.vtk_widget.clear_surface_cache()
        self._update_3d_cache_overlay()

    def _invalidate_shape_cache_for_mask(self, mask):
        """Invalidate cached slice shapes only where a 3D boolean mask changed."""
        if not hasattr(self, "shapeCache") or mask is None:
            return
        if mask.ndim != 3:
            self._invalidate_shape_cache()
            return
        for axis in range(3):
            other_axes = tuple(i for i in range(3) if i != axis)
            affected_slices = np.flatnonzero(np.any(mask, axis=other_axes))
            for slice_index in affected_slices:
                self.shapeCache.pop((axis, int(slice_index)), None)
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            self.vtk_widget.clear_surface_cache()
        self._update_3d_cache_overlay()

    def _invalidate_shape_cache_for_bbox(self, bbox):
        """Invalidate cached slice shapes touched by a 3D bbox with exclusive ends."""
        if not hasattr(self, "shapeCache") or bbox is None:
            return
        z1, z2, y1, y2, x1, x2 = bbox
        axis_ranges = ((z1, z2), (y1, y2), (x1, x2))
        for key in list(self.shapeCache.keys()):
            axis, slice_index = key
            start, end = axis_ranges[axis]
            if start <= slice_index < end:
                self.shapeCache.pop(key, None)
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            self.vtk_widget.clear_surface_cache()
        self._update_3d_cache_overlay()

    def _label_bbox_3d(self, label, padding=0):
        """Return exclusive bbox for a label: (z1, z2, y1, y2, x1, x2)."""
        if not hasattr(self, "tiffMask") or self.tiffMask is None:
            return None
        zs, ys, xs = np.nonzero(self.tiffMask == int(label))
        if zs.size == 0:
            return None
        z1 = max(0, int(zs.min()) - padding)
        z2 = min(self.tiffMask.shape[0], int(zs.max()) + padding + 1)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(self.tiffMask.shape[1], int(ys.max()) + padding + 1)
        x1 = max(0, int(xs.min()) - padding)
        x2 = min(self.tiffMask.shape[2], int(xs.max()) + padding + 1)
        return z1, z2, y1, y2, x1, x2

    def _precache_all_mask_shapes(self):
        """
        Pre-compute and cache mask shapes for all slices in a background thread.
        Merges results into shapeCache on the main thread when done.
        """
        if self.tiffMask is None or self.tiffData is None:
            return
        view_axis = self.currentViewAxis
        num_slices = self.tiffData.shape[view_axis]
        mask_volume = self.tiffMask  # read-only in thread

        def _do_precache():
            results = {}
            for i in range(num_slices):
                key = (view_axis, i)
                results[key] = _compute_shapes_for_slice(mask_volume, view_axis, i)
            return results

        def _on_done(future):
            try:
                results = future.result()
                QtCore.QTimer.singleShot(0, lambda: self._merge_precache_results(results))
            except Exception:
                pass

        executor = getattr(self, "_precache_executor", None) or ThreadPoolExecutor(max_workers=1)
        if not hasattr(self, "_precache_executor"):
            self._precache_executor = executor
        executor.submit(_do_precache).add_done_callback(_on_done)

    def _merge_precache_results(self, results):
        """Merge pre-cached shapes into shapeCache (must run on main thread)."""
        if not hasattr(self, "shapeCache") or results is None:
            return
        self.shapeCache.update(results)
        while len(self.shapeCache) > MAX_SLICE_SHAPE_CACHE:
            self.shapeCache.popitem(last=False)
        self.status(f"Pre-cached mask shapes for {len(results)} slices.")
        self._update_3d_cache_overlay()

    def _update_3d_cache_overlay(self):
        """Update cache counter and solo label overlay on the 3D view."""
        if not hasattr(self, "vtk_widget") or self.vtk_widget is None:
            return
        shape_count = len(self.shapeCache) if hasattr(self, "shapeCache") else 0
        slice_count = len(self.sliceCache) if hasattr(self, "sliceCache") else 0
        solo = None
        if hasattr(self, "visibilityManager") and self.visibilityManager is not None and self.visibilityManager.is_solo_mode():
            solo = self.visibilityManager._solo_label
        self.vtk_widget.update_cache_and_solo_overlay(shape_count, slice_count, solo)

    def _evict_oldest_slice_history(self):
        """Evict oldest slice history from both dicts to prevent unbounded growth."""
        while len(self._undo_history_by_slice) > MAX_SLICES_HISTORY:
            key, _ = self._undo_history_by_slice.popitem(last=False)
            self._mask_history_by_slice.pop(key, None)
        while len(self._mask_history_by_slice) > MAX_SLICES_HISTORY:
            key, _ = self._mask_history_by_slice.popitem(last=False)
            self._undo_history_by_slice.pop(key, None)

    def _touch_slice_history_key(self, key):
        """Move key to end of OrderedDict (most recently used) for LRU."""
        for d in (self._undo_history_by_slice, self._mask_history_by_slice):
            if key in d:
                val = d.pop(key)
                d[key] = val

    def _stash_undo_history(self):
        """Stash current slice undo/redo in background to avoid blocking scroll."""
        if self.tiffData is None:
            return
        key = self._slice_key()
        canvas = self.canvas

        def _do_copy():
            return canvas.copyUndoRedoStacks()

        def _apply(stacks):
            if stacks is None:
                return
            self._touch_slice_history_key(key)
            self._undo_history_by_slice[key] = stacks
            self._evict_oldest_slice_history()

        def _on_done(f):
            try:
                r = f.result()
                QtCore.QTimer.singleShot(0, lambda: _apply(r))
            except Exception:
                pass

        executor = getattr(self, '_stash_executor', None) or ThreadPoolExecutor(max_workers=1)
        if not hasattr(self, '_stash_executor'):
            self._stash_executor = executor
        executor.submit(_do_copy).add_done_callback(_on_done)

    def _restore_undo_history_for_current_slice(self):
        key = self._slice_key()
        if key in self._undo_history_by_slice:
            undo_stack, redo_stack = self._undo_history_by_slice[key]
            self.canvas.setUndoRedoStacks(undo_stack, redo_stack)
        else:
            self.canvas.resetUndoHistory()

    def _get_mask_history(self, key):
        history = self._mask_history_by_slice.get(key)
        if history is None:
            history = {"undo": [], "redo": []}
            self._touch_slice_history_key(key)
            self._mask_history_by_slice[key] = history
            self._evict_oldest_slice_history()
        else:
            self._touch_slice_history_key(key)
        return history

    def _push_mask_undo(self, shape=None):
        """
        Push current mask state to undo. Memory-efficient: stores only the region that will change.
        If shape is provided with a bbox (brush/add/erase), store only that region; else store full slice.
        """
        if self.tiffMask is None:
            return
        key = self._slice_key()
        history = self._get_mask_history(key)
        if shape is not None and hasattr(shape, "points") and len(shape.points) >= 2:
            x1, y1 = shape.points[0].x(), shape.points[0].y()
            x2, y2 = shape.points[1].x(), shape.points[1].y()
            index_tuple = self.get_mask_update_index(
                getattr(shape, "slice_id", self.currentSliceIndex), y1, y2, x1, x2
            )
            old_region = self.tiffMask[index_tuple].copy()
            history["undo"].append(("region", index_tuple, old_region))
        else:
            history["undo"].append(("full", self.get_current_slice(self.tiffMask).copy()))
        if len(history["undo"]) > self.canvas._undo_limit:
            history["undo"].pop(0)
        history["redo"].clear()

    def _apply_mask_entry(self, entry):
        """Apply a mask undo/redo entry (full slice or region)."""
        if self.tiffMask is None:
            return
        if entry[0] == "full":
            current_mask = self.get_current_slice(self.tiffMask)
            current_mask[...] = entry[1]
        else:
            _, index_tuple, old_region = entry
            self.tiffMask[index_tuple] = old_region
        self._invalidate_shape_cache_for_slice()

    def _capture_mask_state_for_redo(self, popped_undo_entry):
        """
        Capture current mask state in the same format as the undo entry, for redo.
        """
        if self.tiffMask is None:
            return None
        if popped_undo_entry[0] == "region":
            _, index_tuple, _ = popped_undo_entry
            return ("region", index_tuple, self.tiffMask[index_tuple].copy())
        return ("full", self.get_current_slice(self.tiffMask).copy())


    def _get_or_create_ai_model(self, model_name):
        """
        Get an AI model instance from cache or create one.
        """
        # If the model is already cached, return it directly
        if model_name in self.ai_model_cache:
            print(f"Loading AI model '{model_name}' from cache.")
            return self.ai_model_cache[model_name]

        # Otherwise, create a new instance
        print(f"Creating new AI model instance: '{model_name}'")
        try:
            # Find the model class
            model_class = [m for m in MODELS if m.name == model_name][0]
            # Create an instance
            model_instance = model_class()
            # Store in cache
            self.ai_model_cache[model_name] = model_instance
            return model_instance
        except IndexError:
            self.errorMessage("Model Not Found", f"The model class for '{model_name}' was not found.")
            return None
        except Exception as e:
            self.errorMessage("Model Creation Error", f"Failed to create model '{model_name}': {e}")
            return None

    def _submit_ai_prompt(self, _) -> None:
        texts = self._ai_prompt_widget.get_text_prompt().split(",")
        boxes, scores, labels = ai.get_rectangles_from_texts(
            model="yoloworld",
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = ai.non_maximum_suppression(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = ai.get_shapes_from_annotations(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        # --- Begin: logic to stop background thread ---
        if self.compute_thread and self.compute_thread.is_alive():
            print("Stopping existing embedding calculation thread...")
            # 1. Set stop event to notify background thread to exit after current loop
            if self.compute_thread_stop_event:
                self.compute_thread_stop_event.set()
            
            # 2. Optionally join with a short timeout (e.g., 2s)
            #self.compute_thread.join(timeout=2.0)
            if self.compute_thread.is_alive():
                print("Warning: Background thread did not stop in time.")
        
        # Reset thread-related variables
        self.compute_thread = None
        self.compute_thread_stop_event = None
        self.embedding_task_queue = None
        # --- End: stop thread logic ---

        # Below is the original reset logic
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.tiffData = None
        self.tiffJsonAnno = None
        self.tiffMask = None
        self.tiffDataLazy = False
        self.sitkImageInfo = None  # NIfTI image metadata (spacing, origin, direction)
        self.annotation_json = None
        self.tiff_mask_file = None
        self.labelFile = None
        self.otherData = None
        self.currentSliceIndex = -1
        self.currentAIPromptPoints = []
        self.embedding_dir = None
        self.current_mask_num = 0
        self.last_ai_mask_slice = 0 # Ensure this is also reset
        self.canvas.resetState()
        self._undo_history_by_slice = collections.OrderedDict()
        self._mask_history_by_slice = collections.OrderedDict()
        self._merge_undo_stack = []
        self._merge_redo_stack = []
        self._watershed_undo_stack = []
        self._watershed_redo_stack = []
        self._label_voxel_counts = {}
        self._labels_in_mask = set()
        self._pending_history_restore_key = None
        self._last_undo_redo = None
        self._mask_autosave_dirty = False
        self._mask_edit_revision = 0
        self._last_autosave_revision = -1
        self._labelJumpInProgress = False
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.camera_initialized = False
            self.vtk_widget.clear_surface_cache()
        self.label_list = [i for i in range(1, MAX_LABEL)]
        self.sliceCache = collections.OrderedDict()
        self.shapeCache = collections.OrderedDict()
        self._slice_scroll_accumulator = 0
        self._slice_scroll_throttle_timer.stop()
        self.lastRendered3DLabel = None  # Reset the last rendered 3D label
        self.toolSwitchedSince3DRender = False  # Reset tool switch tracking

        self._update_3d_cache_overlay()

        # Reset label metadata store
        self.labelMetadataStore.clear()

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """

    def _setCanvasAiModelForMode(self, mode):
        if getattr(self.canvas, "createMode", None) != mode:
            return
        combo = getattr(self, "_selectAiModelComboBox", None)
        if combo is None:
            return
        model = self._get_or_create_ai_model(combo.currentText())
        if model is not None:
            self.canvas.set_ai_model(model, getattr(self, "embedding_dir", None))


    def toggleDrawMode(self, edit=True, createMode="rectangle"):
        draw_actions = {
            "brush": self.actions.createBrushMode,
            "point": self.actions.createPointMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
            "rectangle": self.actions.createBoxAiMaskMode,
            "ai_boundary":self.actions.createAiBoundaryMode,
            "erase": self.actions.createBoxEraseMode,
            "watershed_3d": self.actions.createWatershed3dMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if createMode in {"ai_polygon", "ai_mask", "rectangle", "ai_boundary"}:
            self._setCanvasAiModelForMode(createMode)
        
        # Mark tool switched for 3D re-rendering when switching to an editing tool
        # (not when going back to select/edit mode)
        if not edit:
            self.toolSwitchedSince3DRender = True
        
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def get_mask_update_index(self, slice_id, y1, y2, x1, x2):
        """
        Construct an index tuple for updating self.tiffMask based on the current view axis.
        The original tiffMask shape is assumed to be (D, H, W) corresponding to the original orientation.
        The returned tuple ensures that for the current view, the slice dimension is selected by slice_id,
        and the other two dimensions are updated with y and x values.
        
        For example:
        - Axial (currentViewAxis=0): (slice_id, slice(y1, y2+1), slice(x1, x2+1))
        - Coronal (currentViewAxis=1): (slice(y1, y2+1), slice_id, slice(x1, x2+1))
        - Sagittal (currentViewAxis=2): (slice(y1, y2+1), slice(x1, x2+1), slice_id)
        """
        # Start with a tuple selecting all elements in each axis
        idx = [slice(None)] * 3
        # Insert the slice index into the current view axis
        idx[self.currentViewAxis] = slice_id
        # The remaining axes will be used for y and x.
        remaining_axes = [a for a in range(3) if a != self.currentViewAxis]
        # Assume the first remaining axis corresponds to y and the second to x.
        y_axis = remaining_axes[0]
        x_axis = remaining_axes[1]
        idx[y_axis] = slice(int(y1), int(y2) + 1)
        idx[x_axis] = slice(int(x1), int(x2) + 1)
        return tuple(idx)

    def _update_mask_to_tiffMask(self, shape):
        print("Update mask to tiffMask")
        # Initialize tiffMask if it doesn't exist
        if self.tiffMask is None:
            self.tiffMask = np.zeros(self.tiffData.shape, dtype=np.uint8)
        label = shape.label  # Get the label
        # if label can not convert to int
        if not label.isdigit():
            print(f"input label can not convert to int")
            return
        points = shape.points  # List of points
        mask = shape.mask  # Mask array from shape (should be a binary mask)
        x1, y1 = points[0].x(), points[0].y()
        x2, y2 = points[1].x(), points[1].y()
        print(f"Label: {label}, Slice: {shape.slice_id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, mask shape: {mask.shape}")
        self.current_mask_num = np.sum(mask)
        
        # Construct index tuple based on current view axis and shape coordinates.
        index_tuple = self.get_mask_update_index(shape.slice_id, y1, y2, x1, x2)
        
        affected_labels = []
        
        if self.canvas.createMode == "erase":
            # Find labels that will be affected by erasing
            affected_region = self.tiffMask[index_tuple]
            affected_labels = [l for l in np.unique(affected_region) if l != 0]
            self.tiffMask[index_tuple] = 0
            # For erase mode, we don't need to update the unique label list immediately
            # as removing labels requires full scan anyway (deferred to save operation)
        elif self.canvas.createMode == "brush":
            brush_label = self.brush_label_input.text()
            self.tiffMask[index_tuple][mask > 0] = int(brush_label)
            affected_labels = [brush_label]
            # Fast update: only add the new label to the list
            self.addLabelToUniqueLabelListFast(brush_label)
        else:
            self.tiffMask[index_tuple][mask > 0] = int(label)
            affected_labels = [label]
            # Fast update: only add the new label to the list
            self.addLabelToUniqueLabelListFast(label)
        
        # Mark affected labels as EDITED (user modification)
        if affected_labels:
            self._markLabelsAsEdited(affected_labels)
        
        self._markMaskDirty()
        self.last_ai_mask_slice = shape.slice_id
        self._invalidate_shape_cache_for_slice(shape.slice_id)

    def startAddLabelCompleteTimer(self, shapes):
        """
        Start a timer to trigger the complete addLabel operation after scrolling stops.
        """
        if hasattr(self, "_addLabelTimer"):
            self._addLabelTimer.stop()  # Stop any existing timer

        # Create or reuse a QTimer
        self._addLabelTimer = QTimer(self)
        self._addLabelTimer.setSingleShot(True)  # Trigger only once
        self._addLabelTimer.timeout.connect(lambda: self.executeAddLabelComplete(shapes))
        self._addLabelTimer.start(400)  # Trigger after 600 milliseconds of inactivity


    def executeAddLabelComplete(self, shapes):
        """
        Execute the complete addLabel operation for all shapes.
        """
        for shape in shapes:
            self.addLabelComplete(shape)


    def addLabel(self, shape):
        return
        if shape.label == "0" or shape.label == "10000":
            return
        if not self.enableUpdateLabelList:
            return
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        #self.currentLabelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            rgb = self._get_rgb_by_label(shape.label)
            item = self.uniqLabelList.createItemFromLabel(shape.label, rgb, checked=True)
            self.uniqLabelList.addItem(item)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )
        is_visible = self.label_visibility_states.get(shape.label, True)
        self.canvas.setShapeVisible(shape, is_visible)

        
    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            # 1) Compute the color
            rgb = LABEL_COLORMAP[int(label) % len(LABEL_COLORMAP)]
            # 2) Ensure the list has this label item
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label, rgb=rgb, checked=True)
                self.uniqLabelList.addItem(item)
            return rgb

        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]

        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]

        # fallback
        return (0, 255, 0)

    def _remove_shape_from_mask(self, shape):
        """
        Remove selected shape from mask
        """
        print(f"Remove shape from tiff mask")
        label = shape.label # Get the label
        points = shape.points  # Get the points list
        mask = shape.mask # Get the mask
        x1, y1 = points[0].x(), points[0].y()
        x2, y2 = points[1].x(), points[1].y()

        # Construct an index tuple based on the current view axis.
        index_tuple = self.get_mask_update_index(shape.slice_id, y1, y2, x1, x2)
        self.tiffMask[index_tuple][mask > 0] = 0
        self._markMaskDirty()
        self._invalidate_shape_cache_for_slice(shape.slice_id)

    def addLabelMinimal(self, shape):
        """
        Perform minimal addLabel operations during scrolling.
        """
        self._update_shape_color(shape)  # Only update the shape color
        # Do not set visibility here; handled in bulk in loadShapesFromTiff

    def addLabelComplete(self, shape):
        """
        Perform the complete addLabel operation for shape.
        """
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            rgb = self._get_rgb_by_label(shape.label)
            item = self.uniqLabelList.createItemFromLabel(shape.label, rgb, checked=True)
            self.uniqLabelList.addItem(item)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)

        # Update the shape color
        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )
        # Get visibility from the global state dict; default to True (visible)
        is_visible = self.label_visibility_states.get(shape.label, True)
        self.canvas.setShapeVisible(shape, is_visible)

    def loadShapesFromTiff(self, shapes, replace=True):
        """
        Load shapes with optimized behavior for wheel scrolling and stopping.
        """
        store_history = not getattr(self, "_skip_store_on_next_load", False)
        self._skip_store_on_next_load = False
        if not shapes:  # If there are no shapes, return directly
            if replace:
                self.canvas.loadShapes([], replace=True, store_history=store_history)
            return
            
        self._noSelectionSlot = True

        # Call minimal operation for each shape during scrolling
        for shape in shapes:
            self.addLabelMinimal(shape)

        # Clear selection
        self._noSelectionSlot = False

        # Load shapes into the canvas - this is user-visible; do it immediately
        self.canvas.loadShapes(shapes, replace=replace, store_history=store_history)
        # Tip 4: Skip replaceLastUndoSnapshot on slice change (store_history=False) for speed
        
        # Apply visibility settings immediately using the visibility manager
        # This handles: user checkbox, state-based hiding (e.g., hide verified), and solo mode
        visibility_map = {}
        for shape in shapes:
            label = shape.label
            # Register label with visibility manager if not already registered
            self.visibilityManager.register_label(label)
            # Get effective visibility from visibility manager (combines all visibility factors)
            is_visible = self.visibilityManager.get_effective_visible(label)
            # Also check the legacy label_visibility_states for backward compatibility
            user_checkbox_visible = self.label_visibility_states.get(label, True)
            final_visible = is_visible and user_checkbox_visible
            visibility_map[shape] = final_visible
        
        # Batch update visibility (single canvas redraw)
        if visibility_map:
            self.canvas.setShapesVisible(visibility_map)
        
        # Non-critical UI updates can be deferred
        self.startAddLabelCompleteTimer(shapes)

    def startAddLabelCompleteTimer(self, shapes):
        """
        Start a timer for non-critical UI updates only.
        """
        if hasattr(self, "_addLabelTimer"):
            self._addLabelTimer.stop()

        self._addLabelTimer = QTimer(self)
        self._addLabelTimer.setSingleShot(True)
        self._addLabelTimer.timeout.connect(lambda: self.executeAddLabelCompleteNonCritical(shapes))
        self._addLabelTimer.start(50)  # Significantly reduce delay

    def executeAddLabelCompleteNonCritical(self, shapes):
        """
        Execute only non-critical UI updates that don't affect shape visibility.
        """
        for shape in shapes:
            # Execute only operations that do not affect display
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                rgb = self._get_rgb_by_label(shape.label)
                item = self.uniqLabelList.createItemFromLabel(shape.label, rgb, checked=True)
                self.uniqLabelList.addItem(item)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)
            self.labelDialog.addLabelHistory(shape.label)

    def loadShapes(self, shapes, replace=True):
            self._noSelectionSlot = True

            for shape in shapes:
                self.addLabel(shape)

            self._noSelectionSlot = False
            self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    
    def onUniqLabelItemChanged(self, item: QtWidgets.QListWidgetItem):
        return
        label = item.data(Qt.UserRole)            # String
        visible = (item.checkState() == Qt.Checked)
        
        self.label_visibility_states[label] = visible

        # 1) Shape visibility on Canvas
        # for shape in self.canvas.shapes:
        #     if shape.label == label:
        #         self.canvas.setShapeVisible(shape, visible)

        # 2) Sync items in Polygon Labels list
        #    LabelListWidget is directly iterable and yields QListWidgetItem
        for li in self.labelList:
            if li.shape().label == label:
                li.setCheckState(Qt.Checked if visible else Qt.Unchecked)

        # 3) 3-D view sync (optional)
        try:
            lbl_int = int(label)
            self.vtk_widget.toggle_label_visibility(lbl_int, visible)
        except Exception:
            pass

        self.canvas.update()

    def onUniqLabelVisibilityChanged(self, label: str, visible: bool):
        """Handle label visibility changes in the unique label list (batch update, single redraw)."""
        # 0) Record global state
        self.label_visibility_states[label] = visible

        # 1) Batch-set visibility for shapes on the current slice
        shapes = [s for s in self.canvas.shapes if s.label == label]
        if shapes:
            self.canvas.setShapesVisible({s: visible for s in shapes})  # Single redraw

        # 2) If toggled visible but no shape yet on current slice, create incrementally on demand
        if visible and not shapes and self.tiffMask is not None:
            mask2d = self.get_current_slice(self.tiffMask, self.currentSliceIndex)
            lab = int(label)
            if (mask2d == lab).any():
                y1, y2, x1, x2, roi_mask = self._fast_bbox_and_roi(mask2d, lab)
                shape = Shape(label=str(label), shape_type="mask",
                            description=f"Mask for label {label}",
                            slice_id=self.currentSliceIndex)
                shape.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=roi_mask,
                )
                self.addLabelMinimal(shape)
                self.canvas.loadShapes([shape], replace=False)
                # New shape is visible by default; no need to call setVisible again

        # 3) (Optional) 3D view sync
        try:
            lbl_int = int(label)
            if hasattr(self, 'vtk_widget') and self.vtk_widget:
                self.vtk_widget.toggle_label_visibility(lbl_int, visible)
        except Exception:
            pass

    def _get_slice_range(self, current_index, nextN):
        """
        Generate range for slice indices based on nextN (can be positive or negative).
        
        Args:
            current_index (int): Current slice index.
            nextN (int): Number of slices to predict (positive or negative).
        
        Returns:
            range: Range of slice indices to iterate.
        """
        if nextN > 0:
            # Positive case: From current_index+1 to current_index+nextN
            return range(current_index + 1, current_index + nextN + 1)
        elif nextN < 0:
            # Negative case: From current_index-1 to current_index+nextN (reverse order)
            return range(current_index - 1, current_index + nextN - 1, -1)
        else:
            # nextN is 0, return an empty range
            return range(0)
    
    def predictNextNSlices(self, nextN=5):
        """
        Predict next slices based on current prompt points and AI model.
        
        Args:
            nextN (int): Number of slices to predict (positive or negative).
        """
        print(f"Predicting next {nextN} slices")
        model = self.canvas._ai_model
        
        try:
            for pont_idx, (prompt_point, label) in enumerate(self.currentAIPromptPoints):
                # Calculate the number of mask pixels for the current slice
                self.current_mask_num = np.sum(self.get_current_slice(self.tiffMask) == int(label))
                
                # Get the range of slices to iterate over based on nextN
                slice_range = self._get_slice_range(self.currentSliceIndex, nextN)
                
                for pred_slice_index in slice_range:
                    current_mask = self.get_current_slice(self.tiffMask, pred_slice_index)
                    # Set the current image slice in the AI model
                    model.set_image(
                        self.normalizeImg(
                            self.get_current_slice(self.tiffData, pred_slice_index)
                        ),
                        slice_index=pred_slice_index,
                        embedding_dir=self.embedding_dir,
                    )
                    print(f" Prom point: {prompt_point}, self.canvas.createMode: {self.canvas.createMode}")
                    if self.canvas.createMode == "rectangle":
                        print(f"prompt point: {prompt_point}")
                        # Get mask by box
                        mask = model.predict_mask_from_box(
                            points=prompt_point
                        )
                    elif self.canvas.createMode == "ai_mask":
                        # Get mask by point
                        # Predict the mask from prompt points
                        mask = model.predict_mask_from_points(
                            points=[prompt_point],
                            point_labels=[1],
                        )
                        
                        # Update prompt points based on the predicted mask
                        updated_prompt_points, _ = compute_points_from_mask(mask, original_size=None, use_single_point=True)
                        self.currentAIPromptPoints[pont_idx] = (updated_prompt_points[0], label)                   
                        print(f"Current prompt point: {prompt_point}, Updated prompt points: {updated_prompt_points}")
                    
                    elif self.canvas.createMode == "ai_boundary":
                        # 1. Get the initial filled mask, just like 'ai_mask'
                        full_mask = model.predict_mask_from_points(
                            points=[prompt_point],
                            point_labels=[1],
                        )

                        # 2. Update prompt for the next slice based on the *filled* mask's center
                        if full_mask.any():
                            updated_prompt_points, _ = compute_points_from_mask(full_mask, original_size=None, use_single_point=True)
                            self.currentAIPromptPoints[pont_idx] = (updated_prompt_points[0], label)
                            print(f"Current prompt point: {prompt_point}, Updated prompt points: {updated_prompt_points}")

                            # 3. Convert the filled mask into a 2-pixel boundary
                            eroded_mask = scipy.ndimage.binary_erosion(full_mask)
                            dilated_mask = scipy.ndimage.binary_dilation(full_mask)
                            mask = dilated_mask ^ eroded_mask  # The final mask is now the boundary
                        else:
                            mask = full_mask # If mask is empty, keep it empty

                    if mask is None:
                        continue # Skip if no valid mode was found
                    # Calculate the number of mask pixels in the predicted slice
                    pred_mask_num = np.sum(mask)
                    print(f"Predicting slice {pred_slice_index}, total mask: {pred_mask_num}, label: {label}")
                    
                    # Stop prediction if the predicted mask differs too much from the current mask
                    if abs(pred_mask_num - self.current_mask_num) > 0.2 * self.current_mask_num or current_mask[mask>0].sum() > 0:
                        self.status(f"Stop prediction at slice {pred_slice_index}")
                        break
                    
                    # Update the current mask count and save the mask
                    self.current_mask_num = pred_mask_num
                    self.get_current_slice(self.tiffMask, pred_slice_index)[mask] = int(label)
                    self._markMaskDirty()
        except Exception as e:
            # Catch and print any exception during the process
            print(e)

    def get_current_slice(self, data, slice_id=None):
        """
        Get the current slice from the given data.

        Args:
            data (np.ndarray): The data to get the current slice from.

        Returns:
            np.ndarray: The current slice.
        """
        idx = [slice(None)] * data.ndim
        if slice_id is not None:
            idx[self.currentViewAxis] = slice_id
        else:
            idx[self.currentViewAxis] = self.currentSliceIndex
        return data[tuple(idx)]


    def _get_3d_point_from_2d(self, canvas_pos):
        """
        Convert 2D canvas coordinates and slice index to 3D space (X, Y, Z) based on current view.
        """
        canvas_x = canvas_pos.x()
        canvas_y = canvas_pos.y()
        slice_idx = self.currentSliceIndex

        if self.currentViewAxis == 0:  # Axial view (XY plane)
            # Canvas (x, y) -> 3D (X, Y), slice -> Z
            point_3d = (canvas_x, canvas_y, slice_idx)
        elif self.currentViewAxis == 1:  # Coronal view (XZ plane)
            # Canvas (x, y) -> 3D (X, Z), slice -> Y
            point_3d = (canvas_x, slice_idx, canvas_y)
        elif self.currentViewAxis == 2:  # Sagittal view (YZ plane)
            # Canvas (x, y) -> 3D (Y, Z), slice -> X
            point_3d = (slice_idx, canvas_x, canvas_y)
        else:
            # Default or error case
            point_3d = (0, 0, 0)

        return point_3d


    def get_current_slice_index(self, data):
        """
        Return an index tuple for the current slice of a 3D array `data`,
        based on the current view axis and currentSliceIndex.
        """
        idx = [slice(None)] * data.ndim
        idx[self.currentViewAxis] = self.currentSliceIndex
        return tuple(idx)

    def get_intensity_at(self, pos):
        """
        Attempt to get the intensity at the given position.

        Args:
            pos (QPoint): The position to get the intensity at.

        Returns:
            int: The intensity at the given position, or -1 if not possible.
        """
        if hasattr(self, 'tiffData') and self.tiffData is not None:
            current_slice = self.get_current_slice(self.tiffData)
            x, y = int(pos.x()), int(pos.y())
            if 0 <= y < current_slice.shape[0] and 0 <= x < current_slice.shape[1]:
                return current_slice[y, x]
        return -1
    def get_mask_value_at(self, pos):
        """
        Attempt to get the mask value at the given position.

        Args:
            pos (QPoint): The position to get the mask value at.

        Returns:
            int: The mask value at the given position, or -1 if not possible.
        """
        if hasattr(self, 'tiffMask') and self.tiffMask is not None:
            current_mask = self.get_current_slice(self.tiffMask)
            x, y = int(pos.x()), int(pos.y())
            if 0 <= y < current_mask.shape[0] and 0 <= x < current_mask.shape[1]:
                return current_mask[y, x]
        return -1

    def _resolve_default_label_for_new_mask(self, prompt_points=None):
        """
        Resolve default label for a newly created mask:
        1) label under current position/prompt point
        2) fallback to most recently used label
        """
        candidates = []

        if prompt_points:
            first = prompt_points[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                candidates.append(QtCore.QPointF(float(first[0]), float(first[1])))

        # If prompt point is unavailable (e.g. rectangle mode), use center of the last shape.
        if not candidates and getattr(self.canvas, "shapes", None):
            shape = self.canvas.shapes[-1]
            if getattr(shape, "points", None):
                p1 = shape.points[0]
                p2 = shape.points[1] if len(shape.points) > 1 else shape.points[0]
                candidates.append(
                    QtCore.QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
                )

        if not candidates and self.lastClickedPoint is not None:
            candidates.append(self.lastClickedPoint)

        for pos in candidates:
            label_val = self.get_mask_value_at(pos)
            try:
                label_int = int(label_val)
            except (TypeError, ValueError):
                continue
            if label_int > 0:
                return str(label_int)

        recent = str(getattr(self, "recent_label", "")).strip()
        if recent.isdigit():
            recent_int = int(recent)
            # Ignore background and reserved boundary helper label for normal mask input.
            if recent_int > 0 and recent_int != 10000:
                return recent

        return None

    # Callback functions:
    def newShape(self, prompt_points=None):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        print(f"newShape: {prompt_points}, createMode: {self.canvas.createMode}")
        
        # Use current propmpt points to predict next 5 slices
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self.canvas.createMode == "erase": # 
            text = "0"
        elif self.canvas.createMode == "ai_boundary":
            text = "10000"
        elif self.canvas.createMode == "brush": # if use brush, get brush label
            text = self.brush_label_input.text()
            # if text can not convert to int, return
            if not text.isdigit():
                text = None
                print(f"Brush label can not convert to int: {text}")
        else:
            if self.canvas.createMode in ["ai_mask", "rectangle"]:
                auto_text = self._resolve_default_label_for_new_mask(prompt_points)
                if auto_text:
                    text = auto_text
            if self._config["display_label_popup"] or not text:
                previous_text = self.labelDialog.edit.text()
                text, flags, group_id, description = self.labelDialog.popUp(text)
                if not text:
                    self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            shape = self.canvas.setLastLabel(text, flags)
            if shape is None:
                logger.warning(
                    "newShape skipped: setLastLabel returned None (mode=%s, text=%s, prompt_points=%s, canvas_shapes=%d)",
                    self.canvas.createMode,
                    text,
                    prompt_points,
                    len(getattr(self.canvas, "shapes", [])),
                )
                self.canvas.undoLastLine()
                self.canvas.deleteSelected()
                return
            if prompt_points:
                # Add prompt points to currentAIPromptPoints
                # If createMode is "rectangle", add all prompt points, otherwise add the first prompt point
                if self.canvas.createMode == "rectangle":
                    self.currentAIPromptPoints.append((prompt_points, shape.label))
                elif len(prompt_points) > 0:
                    self.currentAIPromptPoints.append((prompt_points[0], shape.label))
            shape.group_id = group_id
            shape.description = description
            shape.slice_id = self.currentSliceIndex
            self.canvas.replaceLastUndoSnapshot()
            print(f"createMode: {self.canvas.createMode}")
            self.addLabel(shape)
            if shape.shape_type == "mask":
                self._push_mask_undo(shape=shape)
                self._update_mask_to_tiffMask(shape)
                # Refresh current slice with immediate shape loading for brush/erase
                # This avoids the timer delay while still showing the updated mask
                if self.canvas.createMode in ["brush", "erase"]:
                    self.openNextImg(nextN=0, immediate_load=True, store_history=False)
                else:
                    self.openNextImg(nextN=0, store_history=False)
            
            if shape.shape_type == "points": # use these points as the prompt points
                pass
            self.actions.undoLastPoint.setEnabled(False)
            self.setDirty()
            self.recent_label = shape.label  # Store the most recent label for quick access
            # --- Core change: reprioritize embedding calculation tasks ---
            if self.canvas.createMode in ["ai_mask", "ai_boundary", "rectangle"]:
                # Check whether the task queue exists
                if self.embedding_task_queue is not None:
                    self.status("Re-prioritizing embedding calculation...")

                    # 1. Clear all pending tasks in the current queue
                    while not self.embedding_task_queue.empty():
                        try:
                            self.embedding_task_queue.get_nowait()
                        except queue.Empty:
                            break

                    # 2. Generate a new priority list based on the current slice
                    start_index = shape.slice_id
                    num_slices = self.tiffData.shape[self.currentViewAxis]
                    all_indices = list(range(num_slices))
                    prioritized_indices = all_indices[start_index:] + all_indices[:start_index]

                    # 3. Re-add tasks to the queue in the new order
                    for i in prioritized_indices:
                        self.embedding_task_queue.put(i)
        else:
            self.canvas.undoLastLine()
            self.canvas.deleteSelected()
    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        value = max(
            float(self.zoomWidget.minimum()),
            min(float(self.zoomWidget.maximum()), float(value)),
        )
        self._zoom_value_float = value
        self.zoomWidget.setValue(int(round(value)))
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        current = getattr(self, "_zoom_value_float", float(self.zoomWidget.value()))
        spin_value = float(self.zoomWidget.value())
        if abs(current - spin_value) > 1.0:
            current = spin_value
        self.setZoom(current * increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        if not delta:
            return
        # One physical mouse-wheel notch is usually 120 angle-delta units.
        # Trackpads emit many smaller deltas, so use a continuous exponential
        # factor instead of applying a fixed 10% jump per event.
        units = math.pow(1.1, float(delta) / 120.0)
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)


    def normalizeImg(self, img):
        img = np.asarray(img)
        if img.size == 0:
            return np.zeros_like(img, dtype=np.uint8)

        img = img.astype(np.float32, copy=False)
        nonzero = img[img > 0]

        if nonzero.size > 0:
            # Sparse microscopy slices often have a large zero-valued background
            # plus a small bright foreground. Stretching on non-zero percentiles
            # yields a much more usable default view than raw min/max.
            low = float(np.percentile(nonzero, 1.0))
            high = float(np.percentile(nonzero, 99.5))
        else:
            low = float(np.min(img))
            high = float(np.max(img))

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            if nonzero.size == 0:
                return np.zeros_like(img, dtype=np.uint8)
            return (img > 0).astype(np.uint8) * 255

        img = np.clip(img, low, high)
        img = 255.0 * (img - low) / (high - low)
        return img.astype(np.uint8)

    def _load_tiff_volume(self, filename):
        """Load TIFF lazily when the file layout allows memory mapping."""
        try:
            return tiff.memmap(filename), True
        except Exception as exc:
            logger.warning("TIFF memmap unavailable for %s: %s", filename, exc)
            return tiff.imread(filename), False

    def _setCurrentImageFromSlice(self):
        self.imageData = np.ascontiguousarray(
            self.normalizeImg(
                self.get_current_slice(self.tiffData, self.currentSliceIndex)
            )
        )
        h, w = self.imageData.shape
        self.image = QImage(
            self.imageData.data,
            w,
            h,
            self.imageData.strides[0],
            QImage.Format_Grayscale8,
        )

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            self.canvas.setEnabled(True)
            return False

        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))

        # Check if the file is a TIFF file
        if filename.lower().endswith(('.tiff', '.tif')):
            try:
                self.tiffData, self.tiffDataLazy = self._load_tiff_volume(filename)
                file_dir = osp.dirname(filename)
                cell_name = osp.basename(filename).split(".")[0]
                model_name = self._selectAiModelComboBox.currentText()
                self.embedding_dir = f"{file_dir}/{cell_name}_embeddings_{model_name}_axis{self.currentViewAxis}"
                model_instance = self._get_or_create_ai_model(model_name)
                if model_instance:
                    self.canvas.set_ai_model(model_instance, self.embedding_dir)
                self.currentSliceIndex = 0
                if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < self.tiffData.shape[self.currentViewAxis]:
                    self.status("Starting background embedding calculation...")
                    self.embedding_task_queue = queue.Queue()
                    self.compute_thread_stop_event = threading.Event()
                    num_slices = self.tiffData.shape[self.currentViewAxis]
                    for i in range(num_slices):
                        self.embedding_task_queue.put(i)
                    model_name = self._selectAiModelComboBox.currentText()
                    self.compute_thread = threading.Thread(
                        target=compute_tiff_sam_feature,
                        args=(self.tiffData, model_name, self.embedding_dir, self.currentViewAxis, self.embedding_task_queue, self.compute_thread_stop_event),
                        daemon=True
                    )
                    self.compute_thread.start()
                if self.tiffData.ndim == 3:
                    self.imagePath = filename
                    self._setCurrentImageFromSlice()
                else:
                    self.errorMessage(self.tr("Error opening file"), self.tr("Only 3D TIFF files with grayscale slices are supported."))
                    return False
            except Exception as e:
                self.errorMessage(self.tr("Error opening file"), self.tr("Failed to read TIFF file: %s") % str(e))
                return False
        elif filename.lower().endswith(('.nii', '.nii.gz')):
            try:
                sitk_image = sitk.ReadImage(filename)
                self.tiffData = sitk.GetArrayFromImage(sitk_image)
                self.sitkImageInfo = {'spacing': sitk_image.GetSpacing(), 'origin': sitk_image.GetOrigin(), 'direction': sitk_image.GetDirection()}
                nii_spacing = sitk_image.GetSpacing()
                self.spacing_x_input.setText(f"{nii_spacing[0]:.4f}")
                self.spacing_y_input.setText(f"{nii_spacing[1]:.4f}")
                self.spacing_z_input.setText(f"{nii_spacing[2]:.4f}")
                file_dir = osp.dirname(filename)
                base_name = osp.basename(filename)
                cell_name = base_name[:-7] if base_name.lower().endswith('.nii.gz') else base_name.rsplit('.', 1)[0]
                model_name = self._selectAiModelComboBox.currentText()
                self.embedding_dir = f"{file_dir}/{cell_name}_embeddings_{model_name}_axis{self.currentViewAxis}"
                model_instance = self._get_or_create_ai_model(model_name)
                if model_instance:
                    self.canvas.set_ai_model(model_instance, self.embedding_dir)
                self.currentSliceIndex = 0
                if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < self.tiffData.shape[self.currentViewAxis]:
                    self.status("Starting background embedding calculation...")
                    self.embedding_task_queue = queue.Queue()
                    self.compute_thread_stop_event = threading.Event()
                    num_slices = self.tiffData.shape[self.currentViewAxis]
                    for i in range(num_slices):
                        self.embedding_task_queue.put(i)
                    self.compute_thread = threading.Thread(
                        target=compute_tiff_sam_feature,
                        args=(self.tiffData, model_name, self.embedding_dir, self.currentViewAxis, self.embedding_task_queue, self.compute_thread_stop_event),
                        daemon=True
                    )
                    self.compute_thread.start()
                if self.tiffData.ndim == 3:
                    self.imagePath = filename
                    self._setCurrentImageFromSlice()
                else:
                    self.errorMessage(self.tr("Error opening file"), self.tr("Only 3D NIfTI files with grayscale slices are supported."))
                    return False
            except Exception as e:
                self.errorMessage(self.tr("Error opening file"), self.tr("Failed to read NIfTI file: %s") % str(e))
                return False
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData is not None:
                self.imagePath = filename
            self.labelFile = None
            self.image = QImage.fromData(self.imageData) if self.imageData is not None else QImage()
            self.currentSliceIndex = 0

        if self.image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            self.canvas.setEnabled(True)
            return False

        self.canvas.loadPixmap(QPixmap.fromImage(self.image), slice_id=self.currentSliceIndex)
        self.filename = filename

        if filename.lower().endswith('.nii.gz'):
            self.annotation_json = filename[:-7] + ".json"
        elif filename.lower().endswith('.nii'):
            self.annotation_json = filename[:-4] + ".json"
        else:
            self.annotation_json = filename.replace(".tiff", ".json").replace(".tif", ".json")
        if os.path.exists(self.annotation_json):
            try:
                with open(self.annotation_json, "r") as f:
                    self.tiffJsonAnno = json.load(f)
                shapes = []
                slice_key = str(self.currentSliceIndex)
                if slice_key in self.tiffJsonAnno and 'rectangle' in self.tiffJsonAnno[slice_key]:
                    for rect in self.tiffJsonAnno[slice_key]['rectangle']:
                        x1, y1, x2, y2, label = rect
                        shape = Shape(label=label, shape_type="rectangle", description="", slice_id=self.currentSliceIndex)
                        shape.addPoint(QtCore.QPointF(x1, y1))
                        shape.addPoint(QtCore.QPointF(x2, y2))
                        shapes.append(shape)
                self.canvas.storeShapes()
                self.loadShapes(shapes, replace=False)
                self.status(f"Loaded annotations from {self.annotation_json}")
            except Exception as e:
                self.errorMessage(self.tr("Error loading annotations"), self.tr("Failed to read JSON file: %s") % str(e))

        if filename.lower().endswith('.nii.gz'):
            self.tiff_mask_file = filename[:-7] + "_mask.nii.gz"
        elif filename.lower().endswith('.nii'):
            self.tiff_mask_file = filename[:-4] + "_mask.nii.gz"
        else:
            self.tiff_mask_file = filename.replace(".tif", "_mask.tif")
        if self.tiff_mask_file != filename:
            try:
                mask_source = None
                if os.path.exists(self.tiff_mask_file):
                    self.tiffMask = self._readMaskFile(self.tiff_mask_file)
                    mask_source = self.tiff_mask_file

                recovered_source = self._maybeRecoverTempMaskAutosave()
                if recovered_source is not None:
                    mask_source = recovered_source

                if mask_source is not None:
                    self.updateUniqueLabelListFromEntireMask()
                    self._loadLabelMetadata()
                    mask_data = np.ascontiguousarray(
                        self.get_current_slice(self.tiffMask, self.currentSliceIndex)
                    )
                    shapes = _compute_shapes_from_mask_slice(mask_data, self.currentSliceIndex)
                    self.canvas.storeShapes()
                    self.loadShapes(shapes, replace=False)
                    self.status(f"Loaded mask annotations from {mask_source}")
            except Exception as e:
                self.errorMessage(self.tr("Error loading mask file"), self.tr("Failed to read mask file: %s") % str(e))

        if not self.canvas._undo_stack:
            self.canvas.storeShapes()
        self.setClean()
        self.canvas.setEnabled(True)
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        # All-slice shape pre-cache is intentionally opt-in for dense volumes.
        if (
            PRECACHE_ALL_MASK_SHAPES_ON_OPEN
            and hasattr(self, "tiffData")
            and self.tiffData is not None
            and hasattr(self, "tiffMask")
            and self.tiffMask is not None
        ):
            self._precache_all_mask_shapes()
        self._update_3d_cache_overlay()
        return True

    def resizeEvent(self, event):
        zoom_mode = getattr(self, "zoomMode", self.MANUAL_ZOOM)
        if (
            self.canvas
            and not self.image.isNull()
            and zoom_mode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def _ensureWindowVisible(self):
        app = QtWidgets.QApplication.instance()
        if app is None or not hasattr(app, "desktop"):
            return
        desktop = app.desktop()
        if desktop is None:
            return
        screen_num = desktop.screenNumber(self.frameGeometry().center())
        if screen_num < 0:
            screen_num = desktop.primaryScreen()
        if screen_num < 0:
            return

        available = desktop.availableGeometry(screen_num)
        min_w, min_h = 640, 480
        width = max(min_w, min(self.width(), available.width()))
        height = max(min_h, min(self.height(), available.height()))
        if width != self.width() or height != self.height():
            self.resize(width, height)

        frame = self.frameGeometry()
        max_x = available.right() - frame.width() + 1
        max_y = available.bottom() - frame.height() + 1
        x = min(max(frame.x(), available.left()), max_x)
        y = min(max(frame.y(), available.top()), max_y)
        if x != frame.x() or y != frame.y():
            self.move(x, y)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        scalers = getattr(self, "scalers", None)
        if not scalers:
            return
        scaler = scalers.get(self.FIT_WINDOW if initial else self.zoomMode)
        if (
            scaler is None
            or not self.canvas
            or self.canvas.pixmap is None
            or self.canvas.pixmap.isNull()
        ):
            return
        value = scaler()
        value = int(100 * value)
        self._zoom_value_float = float(value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        mask_save_enabled = (
            hasattr(self, "actions")
            and hasattr(self.actions, "saveMask")
            and self.actions.saveMask.isEnabled()
        )
        if getattr(self, "_mask_autosave_dirty", False) or mask_save_enabled:
            self._autosaveTempMask(force=True)
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def eventFilter(self, obj, event):
        """Filter key repeat for Enter to prevent crash when spamming."""
        if event.type() == QtCore.QEvent.KeyPress and event.isAutoRepeat():
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if obj in (self.labelSearchBox, self.uniqLabelList, self.canvas):
                    return True  # Swallow key repeat
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevTenImg(self, _value=False):
        self.openPrevImg(nextN=10)

    def _loadMaskData(self, slice_index, shapes):
        """Load mask data for the specified slice."""
        if self.tiffMask is None:
            return
        mask_data = self.get_current_slice(self.tiffMask, slice_index)
        for shape in _compute_shapes_from_mask_slice(mask_data, slice_index):
            if self.label_visibility_states.get(shape.label, True):
                shapes.append(shape)

    def _fast_bbox_and_roi(self, mask2d: np.ndarray, label: int):
        """Return (y1, y2, x1, x2, roi_mask); faster than imgviz.bboxes."""
        ys, xs = np.where(mask2d == label)  # Get coordinates only; avoid building full boolean image
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        h, w = y2 - y1 + 1, x2 - x1 + 1
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[ys - y1, xs - x1] = True
        return y1, y2, x1, x2, roi_mask

    def updateViewAxis(self, index):
        """
        Update the viewing axis when switching dimensions.
        0 = Axial (default), 1 = Coronal, 2 = Sagittal
        Does NOT reload the file - only refreshes display from in-memory data
        to avoid data loss from resetState.
        """
        if self.tiffData is None and self.filename is None:
            return
        self.currentViewAxis = index
        self.canvas.setCurrentViewAxis(index)  # Update canvas so watershed seeds display correctly
        self.currentSliceIndex = 0  # Reset to the first slice in new view

        # Update embedding_dir for the new view axis (used by AI model)
        if self.filename and self.tiffData is not None:
            file_dir = osp.dirname(self.filename)
            base_name = osp.basename(self.filename)
            if base_name.lower().endswith('.nii.gz'):
                cell_name = base_name[:-7]
            else:
                cell_name = base_name.rsplit('.', 1)[0]
            model_name = self._selectAiModelComboBox.currentText()
            self.embedding_dir = f"{file_dir}/{cell_name}_embeddings_{model_name}_axis{self.currentViewAxis}"
            model_instance = self._get_or_create_ai_model(model_name)
            if model_instance:
                self.canvas.set_ai_model(model_instance, self.embedding_dir)
            # Start background embedding computation for new axis if needed
            if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < self.tiffData.shape[self.currentViewAxis]:
                # Stop previous embedding thread if running
                if hasattr(self, 'compute_thread_stop_event') and self.compute_thread_stop_event is not None:
                    self.compute_thread_stop_event.set()
                self.status("Starting background embedding calculation...")
                self.embedding_task_queue = queue.Queue()
                self.compute_thread_stop_event = threading.Event()
                num_slices = self.tiffData.shape[self.currentViewAxis]
                for i in range(num_slices):
                    self.embedding_task_queue.put(i)
                self.compute_thread = threading.Thread(
                    target=compute_tiff_sam_feature,
                    args=(self.tiffData, model_name, self.embedding_dir, self.currentViewAxis, self.embedding_task_queue, self.compute_thread_stop_event),
                    daemon=True
                )
                self.compute_thread.start()

        self.updateDisplayedSlice()
        self.status(self._current_slice_status_text())
        if hasattr(self, '_sliceLoadTimer') and self.tiffData is not None:
            self._sliceLoadTimer.stop()
            self._sliceLoadTimer.start(self._sliceLoadDelayMs)

    def updateDisplayedSlice(self):
        """
        Update the displayed slice based on the selected view plane.
        Uses a bounded LRU pixmap cache to keep slice loading speed constant over time.
        """
        if self.tiffData is None:
            return

        cache_key = self._slice_key()
        if cache_key in self.sliceCache:
            cached = self.sliceCache.pop(cache_key)
            if isinstance(cached, tuple):
                pixmap, self.imageData = cached
            else:
                pixmap = cached
                self.imageData = np.ascontiguousarray(
                    self.normalizeImg(self.get_current_slice(self.tiffData))
                )
                cached = (pixmap, self.imageData)
            self.sliceCache[cache_key] = cached
            self.canvas.loadPixmap(pixmap, slice_id=self.currentSliceIndex)
        else:
            slice_data = np.ascontiguousarray(
                self.normalizeImg(self.get_current_slice(self.tiffData))
            )
            self.imageData = slice_data
            h, w = slice_data.shape
            bytes_per_line = slice_data.strides[0]
            image = QtGui.QImage(slice_data.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.sliceCache[cache_key] = (pixmap, slice_data)
            while len(self.sliceCache) > MAX_SLICE_PIXMAP_CACHE:
                self.sliceCache.popitem(last=False)
            self.canvas.loadPixmap(pixmap, slice_id=self.currentSliceIndex)

        # Defer VTK crosshair update so slice image appears immediately
        def _update_vtk_crosshair():
            if not hasattr(self, 'tiffData') or self.tiffData is None:
                return
            if self.crosshair_center_xy is None:
                h, w = self.get_current_slice(self.tiffData).shape[:2]
                center_x, center_y = w / 2, h / 2
            else:
                center_x, center_y = self.crosshair_center_xy
            canvas_center_pos = QtCore.QPointF(center_x, center_y)
            point_3d = self._get_3d_point_from_2d(canvas_center_pos)
            try:
                spacing = (float(self.spacing_x_input.text()), float(self.spacing_y_input.text()), float(self.spacing_z_input.text()))
            except (ValueError, AttributeError):
                spacing = (1.0, 1.0, 1.0)
            self.vtk_widget.update_crosshair_position(point_3d, (self.tiffData.shape[2], self.tiffData.shape[1], self.tiffData.shape[0]), spacing=spacing)
        QtCore.QTimer.singleShot(0, _update_vtk_crosshair)
        self._update_3d_cache_overlay()

    def openPrevImg(self, _value=False, load=True, nextN=1):
        """
        Navigate to the previous slice, using cached data if available.
        Automatically trigger caching for surrounding slices.
        """
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if hasattr(self, "tiffData") and self.tiffData is not None:
            self._slice_scroll_accumulator = 0
            self._slice_scroll_throttle_timer.stop()
            # Check if the previous slice exists
            if self.currentSliceIndex - nextN >= 0:
                if nextN != 0:
                    self._stash_undo_history()
                    self._pending_history_restore_key = self._slice_key(self.currentSliceIndex - nextN)
                self.currentSliceIndex -= nextN  # Update to the previous slice index

                self.updateDisplayedSlice()
                # Delay loading annotations and masks
                #QtCore.QTimer.singleShot(0, self.loadAnnotationsAndMasks)
                self._sliceLoadTimer.stop()
                self._sliceLoadTimer.start(self._sliceLoadDelayMs)

                return
            else:
                self.status("Already at the first slice of the TIFF file.")
                return

        # Fallback logic for non-TIFF data
        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex - 1 >= 0:
                filename = self.imageList[currIndex - 1]
            else:
                filename = self.imageList[0]

        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True, nextN=1, immediate_load=False, store_history=True):
        """
        Navigate to the next slice, using cached data if available.
        Automatically trigger caching for surrounding slices.
        
        Parameters:
            immediate_load: If True, load shapes immediately without timer delay (for brush edits)
            store_history: If False, skip pushing this refresh into the undo stack
        """
        if not store_history and hasattr(self, "tiffData") and self.tiffData is not None:
            self._skip_store_on_next_load = True
            if nextN == 0:
                immediate_load = True
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if hasattr(self, "tiffData") and self.tiffData is not None:
            self._slice_scroll_accumulator = 0
            self._slice_scroll_throttle_timer.stop()
            # Check if the next slice exists
            max_slices = self.tiffData.shape[self.currentViewAxis]
            if self.currentSliceIndex + nextN < max_slices:
                if nextN != 0:
                    self._stash_undo_history()
                    self._pending_history_restore_key = self._slice_key(self.currentSliceIndex + nextN)
                self.currentSliceIndex += nextN  # Update to the next slice index
                self.updateDisplayedSlice()

                # For immediate loading (e.g., after brush edits), call directly without timer
                if immediate_load:
                    self._sliceLoadTimer.stop()
                    self.loadAnnotationsAndMasks()
                    return

                # Delay loading annotations and masks
                #QtCore.QTimer.singleShot(0, self.loadAnnotationsAndMasks)
                self._sliceLoadTimer.stop()
                self._sliceLoadTimer.start(self._sliceLoadDelayMs)

                return
            else:
                self.status("Already at the last slice of the TIFF file.")
                return

        # Fallback logic for non-TIFF data
        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]

        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def addLabelToUniqueLabelListFast(self, label_str):
        """
        Quickly add a single label to the unique label list without recalculating
        np.unique on the entire mask. This is much faster for incremental updates.
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return

        label_str = str(label_str)
        self._labels_in_mask.add(label_str)
        if self.uniqLabelList.findItemByLabel(label_str) is None:
            rgb = self._get_rgb_by_label(label_str)
            item = self.uniqLabelList.createItemFromLabel(label_str, rgb=rgb, checked=True)
            self.uniqLabelList.addItem(item)
            self._updateLabelCounter()

    def _syncUniqueLabelListFromCachedStats(self):
        """Update the label list from cached label counts without scanning the mask."""
        self._labels_in_mask = {
            str(label)
            for label, count in self._label_voxel_counts.items()
            if int(count) > 0
        }
        self.uniqLabelList.set_label_voxel_counts(self._label_voxel_counts)

        labels_in_widget = set()
        for i in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(i)
            labels_in_widget.add(item.data(QtCore.Qt.UserRole))

        labels_to_add = self._labels_in_mask - labels_in_widget
        if labels_to_add:
            import natsort

            for label in natsort.natsorted(list(labels_to_add)):
                rgb = self._get_rgb_by_label(label)
                item = self.uniqLabelList.createItemFromLabel(
                    label, rgb=rgb, checked=True
                )
                self.uniqLabelList.addItem(item)

        labels_to_remove = labels_in_widget - self._labels_in_mask
        for label in labels_to_remove:
            item = self.uniqLabelList.findItemByLabel(label)
            if item:
                self.uniqLabelList.takeItem(self.uniqLabelList.row(item))
        self._updateLabelCounter()

    def _applyLabelCountDelta(self, label, delta):
        label = str(label)
        current = int(self._label_voxel_counts.get(label, 0))
        updated = current + int(delta)
        if updated > 0:
            self._label_voxel_counts[label] = updated
        else:
            self._label_voxel_counts.pop(label, None)

    def _updateCachedCountsForMerge(self, source_label, target_label, source_count):
        """
        Update label count/list caches after moving source_count voxels from
        source_label to target_label.
        """
        if not getattr(self, "_label_voxel_counts", None):
            self.updateUniqueLabelListFromEntireMask()
            return
        self._applyLabelCountDelta(source_label, -source_count)
        self._applyLabelCountDelta(target_label, source_count)
        self._syncUniqueLabelListFromCachedStats()
    
    def _updateCachedCountsForRelabel(self, removed_label, removed_count, new_label_counts):
        """Update label count/list caches after replacing one label with new labels."""
        if not getattr(self, "_label_voxel_counts", None):
            self.updateUniqueLabelListFromEntireMask()
            return
        self._applyLabelCountDelta(removed_label, -int(removed_count))
        for label, count in new_label_counts:
            self._applyLabelCountDelta(label, int(count))
        self._syncUniqueLabelListFromCachedStats()

    def updateUniqueLabelListFromEntireMask(self):
        """
        Sync the unique label list based on the entire tiffMask.
        This method adds missing labels and removes labels no longer present in the mask,
        ensuring the list always reflects the full set of labels in the 3D volume.
        
        Note: This is a slow operation for large volumes. Use addLabelToUniqueLabelListFast()
        for incremental updates when only adding labels.
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.uniqLabelList.clear()  # Clear the list if there is no mask
            self._label_voxel_counts = {}
            self._labels_in_mask = set()
            self._updateLabelCounter()
            return

        unique_labels, counts = np.unique(self.tiffMask, return_counts=True)
        self._label_voxel_counts = {
            str(label): int(count)
            for label, count in zip(unique_labels, counts)
            if label != 0
        }
        self._labels_in_mask = set(self._label_voxel_counts)
        self.uniqLabelList.set_label_voxel_counts(self._label_voxel_counts)
        self._syncUniqueLabelListFromCachedStats()

    def loadAnnotationsAndMasks(self):
        """
        Load annotations and masks for the current slice.
        Uses shape cache when available for fast revisits.
        In solo mode, only loads the soloed label for faster display.
        """
        self._skip_store_on_next_load = True
        if self.tiffMask is None:
            self._applyLoadedShapes([], replace=True)
            return
        cache_key = self._slice_key()
        solo_label = (
            self.visibilityManager._solo_label
            if self.visibilityManager.is_solo_mode()
            else None
        )

        if cache_key in self.shapeCache:
            shapes = self.shapeCache.pop(cache_key)
            self.shapeCache[cache_key] = shapes
            # Solo optimization: only pass the soloed label's shapes to canvas
            if solo_label is not None:
                shapes = [s for s in shapes if s.label == solo_label]
            self._applyLoadedShapes(shapes, replace=True)
            self._update_3d_cache_overlay()
            return

        mask_data = np.ascontiguousarray(self.get_current_slice(self.tiffMask))
        shapes = _compute_shapes_from_mask_slice(mask_data, self.currentSliceIndex)
        self.shapeCache[cache_key] = shapes
        while len(self.shapeCache) > MAX_SLICE_SHAPE_CACHE:
            self.shapeCache.popitem(last=False)
        # Solo optimization: only pass soloed label's shapes for faster canvas display
        if solo_label is not None:
            shapes = [s for s in shapes if s.label == solo_label]
        self._applyLoadedShapes(shapes, replace=True)
        self._update_3d_cache_overlay()

    def _applyLoadedShapes(self, shapes, replace=True):
        """Apply loaded shapes to canvas (must run on main thread)."""
        self.loadShapesFromTiff(shapes, replace=replace)
        if self._pending_history_restore_key == self._slice_key():
            self._restore_undo_history_for_current_slice()
            self._pending_history_restore_key = None
        self.setClean()
        self.canvas.setEnabled(True)
        if hasattr(self, 'tiffData') and self.tiffData is not None:
            self.status(self._current_slice_status_text())
        self._updateLabelCounter()
        self._labelJumpInProgress = False

    def _applyScrollAccumulator(self):
        """Apply one step of accumulated scroll (rate-limited for constant slice-change speed)."""
        if not hasattr(self, "tiffData") or self.tiffData is None:
            self._slice_scroll_accumulator = 0
            return
        acc = self._slice_scroll_accumulator
        if acc == 0:
            return
        max_slices = self.tiffData.shape[self.currentViewAxis]
        step = 1 if acc > 0 else -1
        did_move = False
        if step > 0 and self.currentSliceIndex + 1 < max_slices:
            self._slice_scroll_accumulator = acc - step
            did_move = True
            self._stash_undo_history()
            self._pending_history_restore_key = self._slice_key(self.currentSliceIndex + 1)
            self.currentSliceIndex += 1
        elif step < 0 and self.currentSliceIndex - 1 >= 0:
            self._slice_scroll_accumulator = acc - step
            did_move = True
            self._stash_undo_history()
            self._pending_history_restore_key = self._slice_key(self.currentSliceIndex - 1)
            self.currentSliceIndex -= 1
        else:
            self._slice_scroll_accumulator = 0
        if did_move:
            self.updateDisplayedSlice()
            self._sliceLoadTimer.stop()
            # Tip 7: Use longer delay during rapid scroll (accumulator > 2)
            delay = getattr(self, '_sliceLoadDelayMsRapid', 200) if abs(self._slice_scroll_accumulator) > 2 else self._sliceLoadDelayMs
            self._sliceLoadTimer.start(delay)
        if self._slice_scroll_accumulator != 0:
            self._slice_scroll_throttle_timer.start(SLICE_SCROLL_THROTTLE_MS)
    
    def wheelEvent(self, event):
        """
        Mouse wheel event handler. Used to scroll through TIFF slices.
        Throttles rapid scroll events to keep slice-load speed constant and avoid slowdown.
        """
        cursor_pos = QtGui.QCursor.pos()
        scroll_area_pos = self.scrollArea.mapFromGlobal(cursor_pos)
        if event.modifiers() & Qt.ControlModifier:
            angle_delta = event.angleDelta()
            pixel_delta = event.pixelDelta()
            delta_y = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y()
            if delta_y:
                canvas_pos = self.canvas.mapFromGlobal(cursor_pos)
                self.zoomRequest(float(delta_y), canvas_pos)
                event.accept()
                return
        if hasattr(self, "tiffData") and self.tiffData is not None and self.scrollArea.rect().contains(scroll_area_pos):
            # Accumulate scroll direction; apply at fixed rate via _applyScrollAccumulator
            delta = event.angleDelta().y()
            if delta > 0:
                self._slice_scroll_accumulator += 1
            elif delta < 0:
                self._slice_scroll_accumulator -= 1
            if not self._slice_scroll_throttle_timer.isActive():
                self._slice_scroll_throttle_timer.start(SLICE_SCROLL_THROTTLE_MS)
            event.accept()
        else:
            event.ignore()

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        # Get the parent directory of the current working directory
        current_path = os.getcwd()  # Get the current working directory
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))  # Get the parent directory
        
        # Use the directory of the filename if available; otherwise, use the parent directory
        path = osp.dirname(str(self.filename)) if self.filename else parent_path
        
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        # Add support for TIFF and NIfTI 3D image formats
        formats.extend(["*.tif", "*.tiff", "*.nii", "*.nii.gz"])
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)  # Set the default directory to the parent directory
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

    def _markMaskDirty(self):
        if hasattr(self, "actions") and hasattr(self.actions, "saveMask"):
            self.actions.saveMask.setEnabled(True)
        self._mask_autosave_dirty = True
        self._mask_edit_revision = getattr(self, "_mask_edit_revision", 0) + 1

    def _mask_autosave_path(self):
        if not getattr(self, "tiff_mask_file", None):
            return None
        mask_file = self.tiff_mask_file
        lower = mask_file.lower()
        if lower.endswith(".nii.gz"):
            return mask_file[:-7] + ".autosave.nii.gz"
        root, ext = osp.splitext(mask_file)
        return root + ".autosave" + ext

    def _mask_autosave_tmp_path(self, autosave_path):
        if autosave_path.lower().endswith(".nii.gz"):
            return autosave_path[:-7] + ".tmp.nii.gz"
        root, ext = osp.splitext(autosave_path)
        return root + ".tmp" + ext

    def _readMaskFile(self, path):
        if path.lower().endswith((".nii", ".nii.gz")):
            sitk_mask = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(sitk_mask).astype(np.uint16)
        return tiff.imread(path).astype(np.uint16)

    def _writeMaskFile(self, path):
        if path.lower().endswith((".nii", ".nii.gz")):
            sitk_mask = sitk.GetImageFromArray(self.tiffMask)
            if hasattr(self, "sitkImageInfo") and self.sitkImageInfo:
                sitk_mask.SetSpacing(self.sitkImageInfo["spacing"])
                sitk_mask.SetOrigin(self.sitkImageInfo["origin"])
                sitk_mask.SetDirection(self.sitkImageInfo["direction"])
            sitk.WriteImage(sitk_mask, path)
        else:
            tiff.imwrite(path, self.tiffMask, compression="zlib")

    def _writeMaskFileAtomic(self, path):
        tmp_path = self._mask_autosave_tmp_path(path)
        try:
            mask_dir = osp.dirname(path)
            if mask_dir:
                os.makedirs(mask_dir, exist_ok=True)
            self._writeMaskFile(tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if osp.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def _autosaveTempMask(self, force=False):
        if getattr(self, "tiffMask", None) is None or not getattr(self, "tiff_mask_file", None):
            return False
        if not force and not getattr(self, "_mask_autosave_dirty", False):
            return False

        revision = getattr(self, "_mask_edit_revision", 0)
        if not force and self._last_autosave_revision == revision:
            return False

        autosave_path = self._mask_autosave_path()
        if not autosave_path:
            return False

        try:
            self._writeMaskFileAtomic(autosave_path)
            self._last_autosave_revision = revision
            self._mask_autosave_dirty = False
            self.status(
                f"Autosaved temporary mask to {osp.basename(autosave_path)}",
                delay=3000,
            )
            return True
        except Exception as e:
            logger.warning("Failed to autosave temporary mask %s: %s", autosave_path, e)
            self.status(f"Failed to autosave temporary mask: {e}", delay=5000)
            return False

    def _cleanupTempMaskAutosave(self):
        autosave_path = self._mask_autosave_path()
        if not autosave_path:
            return
        for path in (autosave_path, self._mask_autosave_tmp_path(autosave_path)):
            try:
                if osp.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning("Failed to remove temporary mask autosave %s: %s", path, e)

    def _maybeRecoverTempMaskAutosave(self):
        autosave_path = self._mask_autosave_path()
        if not autosave_path or not osp.exists(autosave_path):
            return None

        mask_exists = osp.exists(self.tiff_mask_file)
        if mask_exists and osp.getmtime(autosave_path) <= osp.getmtime(self.tiff_mask_file):
            return None

        if mask_exists:
            detail = (
                f"A newer temporary mask autosave was found:\n{autosave_path}\n\n"
                "Load it instead of the saved mask?"
            )
        else:
            detail = (
                "A temporary mask autosave was found, but the saved mask is missing:\n"
                f"{autosave_path}\n\nLoad the autosaved mask?"
            )
        answer = QtWidgets.QMessageBox.question(
            self,
            "Recover Autosaved Mask",
            detail,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return None

        self.tiffMask = self._readMaskFile(autosave_path)
        if hasattr(self, "actions") and hasattr(self.actions, "saveMask"):
            self.actions.saveMask.setEnabled(True)
        self._mask_autosave_dirty = False
        self._mask_edit_revision = getattr(self, "_mask_edit_revision", 0) + 1
        self._last_autosave_revision = self._mask_edit_revision
        return autosave_path

    def saveMask(self, _value=False):
        """
        Update the mask in a TIFF or NIfTI file using information from an updated JSON file.
        Also saves label metadata to a sidecar JSON file.
        """
        print("save mask")
        self._writeMaskFileAtomic(self.tiff_mask_file)
        if self.tiff_mask_file.lower().endswith(('.nii', '.nii.gz')):
            print(f"Updated NIfTI mask file saved to {self.tiff_mask_file}")
        else:
            print(f"Updated TIFF mask file saved to {self.tiff_mask_file}")
        
        # Save label metadata to sidecar JSON file
        self._saveLabelMetadata()
        self._cleanupTempMaskAutosave()
        self._mask_autosave_dirty = False
        self._last_autosave_revision = getattr(self, "_mask_edit_revision", 0)
        
        self.actions.saveMask.setEnabled(False)
        self.currentAIPromptPoints = []

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename


    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def segmentAll(self):
        print(f"Segmenting all in current slice {self.currentSliceIndex} using model {self._segmentallComboBox.currentText()}")
        if not hasattr(self, 'tiffData') or self.tiffData is None or not hasattr(self, 'imageData') or self.imageData is None:
            print("No image data available.")
            return
        if not getattr(self, "_segment_all_available", True):
            self.errorMessage(
                self.tr("Model unavailable"),
                self.tr("No segment-all model is available in this build."),
            )
            return
        model_name = self._segmentallComboBox.currentText()
        if not hasattr(self, 'segmentAllModel') or self.segmentAllModel is None or self.segmentAllModel.name != model_name:
            candidates = [model for model in MODELS if model.name.lower() == model_name.lower()]
            if not candidates:
                self.errorMessage(
                    self.tr("Model unavailable"),
                    self.tr("Segmentation model '{}' is not available in this build.").format(model_name),
                )
                return
            self.segmentAllModel = candidates[0]()
        pred_mask = self.segmentAllModel.predict(self.imageData)
        if pred_mask is None:
            self.errorMessage(
                self.tr("Segmentation failed"),
                self.tr("Model '{}' did not return a valid mask.").format(model_name),
            )
            return
        # Get the index tuple for the current slice using dynamic slicing.
        idx = self.get_current_slice_index(self.tiffMask)
        if self.tiffMask is None and self.tiffData is not None:
            self.tiffMask = np.zeros(self.tiffData.shape, dtype=np.uint16)
        self.tiffMask[idx] = pred_mask

        # Set save mask button enabled
        self._markMaskDirty()
        self.updateUniqueLabelListFromEntireMask()
        
        # Register new labels with PROPOSED state (from AI segmentation)
        new_labels = [l for l in np.unique(pred_mask) if l != 0]
        self._registerAutoSegmentationLabels(new_labels, LabelOrigin.AI)
        
        # Load shapes in nvas
        shapes = []
        self._loadMaskData(self.currentSliceIndex, shapes)
        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setClean()



    def _compute_center_point(self):
        """
        Compute center point of all masks from current slice and add to current prompt point
        """
        print(f"Compute center point of all masks from current slice {self.currentSliceIndex}")
        if self.tiffMask is None:
            return
        # Reset the prompt point
        self.currentAIPromptPoints = []
        mask = self.get_current_slice(self.tiffMask)
        if np.sum(mask) == 0:
            return
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            # Get the binary mask for the current label
            binary_mask = mask == label

            # Calculate the center of mass
            centroid = measurements.center_of_mass(binary_mask)
        
            # Check if the centroid lies inside the region
            centroid_int = tuple(map(int, centroid))  # Convert to integer index
            if (
                0 <= centroid_int[0] < mask.shape[0] and  # Check within bounds
                0 <= centroid_int[1] < mask.shape[1] and
                binary_mask[centroid_int]  # Check if inside the region
            ):
                centroid = centroid  # Use the original centroid
            else:
                # Find all points in the region
                region_points = np.column_stack(np.where(binary_mask))
                
                # Calculate the distance from the centroid to all region points
                distances = cdist([centroid], region_points)
                
                # Find the closest point in the region
                closest_point = region_points[np.argmin(distances)]
                centroid = closest_point  # Use the closest point as the centroid

            # Add the center point to the prompt point
            self.currentAIPromptPoints.append(((int(centroid[1]), int(centroid[0])), str(label)))


    def pointSelectionChanged(self, point):
        """
        Triggered when the user clicks on the canvas.
        Update crosshair to follow the clicked point in real-time.
        """
        self.lastClickedPoint = point

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return
            
        self.crosshair_center_xy = (point.x(), point.y())

        # Get spacing values from input fields
        try:
            spacing_x = float(self.spacing_x_input.text())
            spacing_y = float(self.spacing_y_input.text())
            spacing_z = float(self.spacing_z_input.text())
            spacing = (spacing_x, spacing_y, spacing_z)
        except (ValueError, AttributeError):
            spacing = (1.0, 1.0, 1.0)
        
        # Use the clicked point directly for real-time update
        point_3d = self._get_3d_point_from_2d(point)
        
        # Update crosshair in 3D view at the clicked position
        # Note: self.tiffData.shape order is (D, H, W) -> (Z, Y, X)
        # while vtk_widget expects (X, Y, Z)
        self.vtk_widget.update_crosshair_position(point_3d, (self.tiffData.shape[2], self.tiffData.shape[1], self.tiffData.shape[0]), spacing=spacing)

        # If single-label rendering mode is active, check if we need to re-render
        if not self.showAll3D:
            # Get the current clicked label
            current_label = self.get_mask_value_at(point)
            # Re-render if:
            # 1. The label has changed, OR
            # 2. The same label was clicked but tool was switched (indicating possible edit)
            label_changed = (current_label != self.lastRendered3DLabel)
            tool_was_switched = self.toolSwitchedSince3DRender
            
            if label_changed or tool_was_switched:
                self.update3D()
                self.lastRendered3DLabel = current_label
                # Clear the tool switch flag after re-rendering
                self.toolSwitchedSince3DRender = False
            
        # Apply spacing to point for camera
        point_3d_scaled = (point_3d[0] * spacing[0], point_3d[1] * spacing[1], point_3d[2] * spacing[2])
        
        # Move the 3D camera focus to the clicked point
        self.vtk_widget.center_camera_on_point(point_3d_scaled)

    def on3DRenderingCheckBoxChanged(self, state: int):
        """
        Handle checkbox state changes:
        - True: render all labels in 3D
        - False: render only the label at the last clicked canvas point
        """
        self.showAll3D = (state == QtCore.Qt.Checked)
        # Reset the last rendered label to force re-rendering when mode changes
        self.lastRendered3DLabel = None
        # Immediately refresh the 3D view
        self.update3D()
    
    def update3D(self):
        """
        Update the 3D view based on showAll3D flag:
        - If True: render the full mask volume
        - If False: render only the mask for the last clicked label
        
        The volume is downsampled by 1/2 in each dimension for faster rendering.
        """
        self.status("Updating 3D view of segmentation")
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            print("No mask data available.")
            self.vtk_widget.update_label_overlay(0, None)
            self._update_3d_cache_overlay()
            return

        if self.showAll3D:
            volume = self.tiffMask
            labels_to_render = None
            # Clear tool switch flag since we're rendering everything
            self.toolSwitchedSince3DRender = False
        else:
            # guard against no point selected yet
            if self.lastClickedPoint is None:
                print("No point selected yet for single-label rendering.")
                return
            # Get the label at the last clicked canvas location
            label = self.get_mask_value_at(self.lastClickedPoint)
            if label <= 0:
                print("Clicked point is background or invalid.")
                return
            # Update the last rendered label
            self.lastRendered3DLabel = label
            # Clear the tool switch flag after re-rendering
            self.toolSwitchedSince3DRender = False
            volume = self.tiffMask
            labels_to_render = {int(label)}

        # Get spacing values from input fields
        try:
            spacing_x = float(self.spacing_x_input.text())
            spacing_y = float(self.spacing_y_input.text())
            spacing_z = float(self.spacing_z_input.text())
            spacing = (spacing_x, spacing_y, spacing_z)
        except ValueError:
            print("Invalid spacing values, using default (1, 1, 1)")
            spacing = (1.0, 1.0, 1.0)

        # Downsample the volume by 1/2 in each dimension for faster rendering
        # Use order=0 (nearest-neighbor) to preserve integer label values
        downsample_factor = 0.5
        volume_downsampled = scipy.ndimage.zoom(
            volume, 
            zoom=downsample_factor, 
            order=0,  # nearest-neighbor interpolation to preserve label values
            mode='nearest'
        )
        print(f"Original volume shape: {volume.shape}, Downsampled shape: {volume_downsampled.shape}")
        
        # Adjust spacing to account for downsampling (multiply by 2)
        spacing_adjusted = (
            spacing[0] / downsample_factor,
            spacing[1] / downsample_factor,
            spacing[2] / downsample_factor
        )

        # Call the existing VTK update routine with adjusted spacing
        smooth_iterations = 10 if self.showAll3D else 20
        self.vtk_widget.update_surface_with_smoothing(
            volume_downsampled,
            smooth_iterations=smooth_iterations,
            spacing=spacing_adjusted,
            labels_to_render=labels_to_render,
        )
        total_count = self.uniqLabelList.count()
        current_label = None if self.showAll3D else self.lastRendered3DLabel
        self.vtk_widget.update_label_overlay(total_count, current_label)
        self._update_3d_cache_overlay()
        self.status("3D view updated.")

    def tracking(self):
        self.status("Checking requirements for tracking...")

        # 1. --- Check and compute embedding features ---
        if self.embedding_dir and self.tiffData is not None:
            num_slices_in_view = self.tiffData.shape[self.currentViewAxis]

            # Check if embeddings need to be computed or completed
            if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < num_slices_in_view:
                self.status("Embedding calculation required. Starting background process...")
                QtWidgets.QApplication.processEvents()  # Force UI refresh to show status

                # Use the recorded "last edited slice" as the start index
                start_index = self.last_ai_mask_slice

                # Start a background thread to compute embeddings from the start index
                model_name = self._selectAiModelComboBox.currentText()
                compute_thread = threading.Thread(
                    target=compute_tiff_sam_feature,
                    args=(self.tiffData, model_name, self.embedding_dir, self.currentViewAxis, start_index),
                    daemon=True
                )
                compute_thread.start()

                # --- Show wait cursor and wait for computation to finish ---
                # Tracking requires all embeddings to be ready
                self.status(f"Calculating embeddings from slice {start_index}... Please wait.")
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

                # Wait for the background thread to finish
                compute_thread.join() 

                QtWidgets.QApplication.restoreOverrideCursor()
                self.status("Embedding calculation complete. Starting tracking.")

        # 2. --- Perform the original tracking logic ---
        self._compute_center_point()  # Requires embeddings to exist

        # Track forward
        self.predictNextNSlices(nextN=100)

        # Track backward
        if self.currentSliceIndex > 0:
            self.predictNextNSlices(nextN=-100)

    def _on_label_opacity_changed(self, value):
        """Update label transparency from slider (0-100 -> 0.0-1.0)."""
        Shape.label_opacity = value / 100.0
        if hasattr(self.canvas, "invalidateMaskOverlay"):
            self.canvas.invalidateMaskOverlay()
        self.canvas.update()

    def merge_labels(self):
        try:
            label1 = int(self.merge_label_input_1.text())
            label2 = int(self.merge_label_input_2.text())
            if not hasattr(self, 'tiffMask') or self.tiffMask is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available.")
                return
            if label1 == label2:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input", "Source and target labels are the same."
                )
                return

            # One full-volume scan: keep the source mask for undo and reuse it
            # for relabeling, cache invalidation, and count updates.
            mask1 = self.tiffMask == label1
            source_count = int(np.count_nonzero(mask1))
            if source_count == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input", f"Label {label1} is not present."
                )
                return
            self._merge_undo_stack.append((label1, label2, mask1))
            if len(self._merge_undo_stack) > MERGE_UNDO_LIMIT:
                self._merge_undo_stack.pop(0)
            self._merge_redo_stack.clear()

            self.tiffMask[mask1] = label2
            self._markMaskDirty()
            self._invalidate_shape_cache_for_mask(mask1)

            # Update metadata: merge source labels into target
            self.labelMetadataStore.handle_merge(
                source_labels=[str(label1)],
                target_label=str(label2),
                target_mask=None,
                push_undo=True
            )
            
            self._updateCachedCountsForMerge(label1, label2, source_count)
            self._updateLabelStateStats()

            # Refresh current slice
            self.openNextImg(nextN=0, store_history=False)
            QtWidgets.QMessageBox.information(
                self, "Success", f"Label {label1} merged into {label2}."
            )
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Enter valid integer labels.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def delete_label(self):
        """
        Deletes the specified label from the mask by setting it to 0.
        """
        try:
            # Get the label to delete from the input field
            label_to_delete = int(self.label_input.text())

            # Check if the tiffMask exists
            if not hasattr(self, 'tiffMask') or self.tiffMask is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available.")
                return

            # Set all values in the mask equal to the label to 0
            self.tiffMask[self.tiffMask == label_to_delete] = 0
            self._markMaskDirty()
            self._invalidate_shape_cache()
            self.updateUniqueLabelListFromEntireMask()

            # Refresh current slice
            self.openNextImg(nextN=0, store_history=False)
            QtWidgets.QMessageBox.information(self, "Success", f"Label {label_to_delete} deleted.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer label.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    # ============== Label Lifecycle Management Methods ==============
    
    def _getSelectedLabel(self) -> str:
        """Get the currently selected label from uniqLabelList."""
        selected_items = self.uniqLabelList.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].data(Qt.UserRole)
    
    def verifySelectedLabel(self):
        """Verify the currently selected label."""
        label = self._getSelectedLabel()
        if label:
            self.verifyLabel(label)
        else:
            self.status("No label selected to verify")
    
    def revertSelectedLabel(self):
        """Revert the currently selected label to its proposed state."""
        label = self._getSelectedLabel()
        if label:
            self.revertLabelToProposed(label)
        else:
            self.status("No label selected to revert")
    
    def rejectSelectedLabel(self):
        """Reject (delete) the currently selected label."""
        label = self._getSelectedLabel()
        if label:
            self.rejectLabel(label)
        else:
            self.status("No label selected to reject")
    
    def verifyLabel(self, label: str):
        """
        Mark a label as VERIFIED by the user.
        """
        label = str(label)
        current_state = self.labelMetadataStore.get_state(label)
        
        if current_state == LabelState.VERIFIED:
            self.status(f"Label {label} is already verified")
            return
        
        self.labelMetadataStore.verify_label(label, push_undo=True)
        self.uniqLabelList.update_label_state(label)
        self._updateLabelStateStats()
        self.status(f"Label {label} verified ✓")
        self.setDirty()
        # Auto-save immediately so verified state persists when reopening the file
        if (
            hasattr(self, "tiff_mask_file")
            and self.tiff_mask_file
            and hasattr(self, "tiffMask")
            and self.tiffMask is not None
        ):
            self.saveMask()
    
    def unverifyLabel(self, label: str):
        """
        Unverify a label (VERIFIED -> EDITED).
        """
        label = str(label)
        current_state = self.labelMetadataStore.get_state(label)
        
        if current_state != LabelState.VERIFIED:
            self.status(f"Label {label} is not verified")
            return
        
        self.labelMetadataStore.unverify_label(label, push_undo=True)
        self.uniqLabelList.update_label_state(label)
        self._updateLabelStateStats()
        self.status(f"Label {label} unverified (now EDITED)")
        self.setDirty()
        # Auto-save immediately so state persists when reopening the file
        if (
            hasattr(self, "tiff_mask_file")
            and self.tiff_mask_file
            and hasattr(self, "tiffMask")
            and self.tiffMask is not None
        ):
            self.saveMask()
    
    def _onCanvasContextMenuAboutToShow(self, pos):
        """Store the canvas position when right-click context menu is about to show."""
        self._lastCanvasContextMenuPos = pos
    
    def verifyLabelAtCursor(self):
        """Verify the label at the last right-click position on the slice canvas."""
        if not hasattr(self, '_lastCanvasContextMenuPos') or self._lastCanvasContextMenuPos is None:
            self.status("Right-click on a label in the slice view first")
            return
        label_val = self.get_mask_value_at(self._lastCanvasContextMenuPos)
        if label_val <= 0:
            self.status("No label at cursor position (click on a labeled region)")
            return
        self.verifyLabel(str(label_val))
    
    def unverifyLabelAtCursor(self):
        """Unverify the label at the last right-click position on the slice canvas."""
        if not hasattr(self, '_lastCanvasContextMenuPos') or self._lastCanvasContextMenuPos is None:
            self.status("Right-click on a label in the slice view first")
            return
        label_val = self.get_mask_value_at(self._lastCanvasContextMenuPos)
        if label_val <= 0:
            self.status("No label at cursor position (click on a labeled region)")
            return
        self.unverifyLabel(str(label_val))

    def soloLabelAtCursor(self):
        """Solo the label at the last right-click position on the slice canvas."""
        if not hasattr(self, '_lastCanvasContextMenuPos') or self._lastCanvasContextMenuPos is None:
            self.status("Right-click on a label in the slice view first")
            return
        label_val = self.get_mask_value_at(self._lastCanvasContextMenuPos)
        if label_val <= 0:
            self.status("No label at cursor position (click on a labeled region)")
            return
        self._onSoloCurrentRequested(str(label_val))
    
    def rejectLabel(self, label: str):
        """
        Reject (delete) a label from the mask.
        This sets all voxels with this label to 0 and removes the label from metadata.
        """
        label = str(label)
        
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available.")
            return
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Reject Label",
            f"Are you sure you want to reject (delete) label {label}?\nThis will remove all voxels with this label.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        try:
            label_int = int(label)
            
            # Push undo for mask changes
            self._push_mask_undo()
            
            # Store metadata for potential undo
            removed_metadata = self.labelMetadataStore.remove(label, push_undo=True)
            
            # Delete from mask
            self.tiffMask[self.tiffMask == label_int] = 0
            self._markMaskDirty()
            self._invalidate_shape_cache()

            # Update UI
            self.updateUniqueLabelListFromEntireMask()
            self._updateLabelStateStats()
            
            # Refresh display
            self.openNextImg(nextN=0, store_history=False)
            
            self.status(f"Label {label} rejected and deleted")
            self.setDirty()
            
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Invalid label format.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def revertLabelToProposed(self, label: str):
        """
        Revert a label to its original proposed state (restore the snapshot).
        """
        label = str(label)
        
        if not self.labelMetadataStore.can_revert(label):
            self.status(f"Label {label} has no proposed snapshot to revert to")
            return
        
        current_state = self.labelMetadataStore.get_state(label)
        if current_state == LabelState.PROPOSED:
            self.status(f"Label {label} is already in PROPOSED state")
            return
        
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available.")
            return
        
        # Push undo for mask changes
        self._push_mask_undo()
        
        # Get the proposed snapshot
        proposed_mask = self.labelMetadataStore.revert_to_proposed(label, push_undo=True)
        
        if proposed_mask is not None:
            try:
                label_int = int(label)
                
                # Clear current label from mask
                self.tiffMask[self.tiffMask == label_int] = 0
                
                # Restore the proposed snapshot
                # The snapshot shape should match a region of the mask
                # For 3D masks, we need to apply it to the correct location
                self.tiffMask[proposed_mask > 0] = label_int
                
                self._markMaskDirty()
                self._invalidate_shape_cache()
                
                # Update UI
                self.updateUniqueLabelListFromEntireMask()
                self.uniqLabelList.update_label_state(label)
                self._updateLabelStateStats()
                
                # Refresh display
                self.openNextImg(nextN=0, store_history=False)
                
                self.status(f"Label {label} reverted to proposed state")
                self.setDirty()
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to revert: {str(e)}")
        else:
            self.status(f"Failed to retrieve proposed snapshot for label {label}")
    
    def commitChanges(self):
        """
        Commit the current working mask to the final mask store.
        This saves a snapshot and records the commit revision.
        Does NOT erase proposed snapshots.
        """
        # Debounce: ignore rapid repeat (e.g. spamming Ctrl+Enter) to prevent crash
        if getattr(self, "_commitCooldown", False):
            return
        self._commitCooldown = True
        QTimer.singleShot(500, lambda: setattr(self, "_commitCooldown", False))

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.status("No mask data to commit")
            return

        revision = self.labelMetadataStore.commit(self.tiffMask)
        
        # Auto-save the mask
        self.saveMask()
        
        # Save metadata
        self._saveLabelMetadata()
        
        self._updateLabelStateStats()
        self.status(f"Changes committed (revision {revision})")
    
    def _updateLabelStateStats(self):
        """Update the label state statistics display."""
        stats = self.labelMetadataStore.get_stats()
        self.labelStateStatsLabel.setText(
            f"○{stats['proposed']} ◐{stats['edited']} ●{stats['verified']}"
        )
        self._updateLabelCounter()
    
    def _updateLabelCounter(self):
        """Update the label counter display (Label X of Y) showing total labels."""
        if not hasattr(self, 'labelCounterLabel'):
            return
        # Get total from mask (source of truth) when available, else from list
        if hasattr(self, 'tiffMask') and self.tiffMask is not None:
            total = len(getattr(self, "_labels_in_mask", set()))
            if total == 0 and self.uniqLabelList.count() > 0:
                total = self.uniqLabelList.count()
        else:
            total = self.uniqLabelList.count()
        if total == 0:
            self.labelCounterLabel.setText("0 labels")
            return
        current_row = self.uniqLabelList.currentRow()
        if current_row >= 0:
            self.labelCounterLabel.setText(f"Label {current_row + 1} of {total}")
        else:
            self.labelCounterLabel.setText(f"{total} labels total")
    
    def _markLabelsAsEdited(self, affected_labels: list):
        """
        Mark labels as EDITED when they are modified by user actions.
        Called after mask editing operations.
        """
        for label in affected_labels:
            label_str = str(label)
            if label_str in self.labelMetadataStore:
                self.labelMetadataStore.mark_edited(label_str)
                self.uniqLabelList.update_label_state(label_str)
        self._updateLabelStateStats()
    
    def _registerAutoSegmentationLabels(self, labels: list, origin: LabelOrigin, store_snapshots=True):
        """
        Register labels created by auto-segmentation with PROPOSED state.
        Stores the current mask as the proposed snapshot for each label.
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return
        
        for label in labels:
            label_str = str(label)
            if label_str == "0":
                continue
            
            label_mask = None
            if store_snapshots:
                # Full-volume snapshots are useful for AI labels but expensive for
                # multi-region watershed outputs.
                label_mask = (self.tiffMask == int(label)).astype(np.uint8)
            
            # Register with metadata store
            self.labelMetadataStore.create_from_auto_segmentation(
                label_str, label_mask, origin
            )
            self.uniqLabelList.update_label_state(label_str)
        
        self._updateLabelStateStats()
    
    def _saveLabelMetadata(self):
        """Save label metadata to a sidecar JSON file (including visibility settings)."""
        if self.tiff_mask_file:
            metadata_path = LabelMetadataStore.get_sidecar_path(self.tiff_mask_file)
            try:
                # Save label metadata store
                self.labelMetadataStore.save(metadata_path)
                print(f"Label metadata saved to {metadata_path}")
                
                # Save visibility settings to a separate section of the same file
                # or a parallel file
                self._saveVisibilitySettings(metadata_path)
            except Exception as e:
                print(f"Failed to save label metadata: {e}")
    
    def _saveVisibilitySettings(self, base_path: str):
        """Save visibility manager settings."""
        import json
        visibility_path = base_path.replace('.json', '_visibility.json')
        try:
            visibility_data = self.visibilityManager.to_dict()
            with open(visibility_path, 'w') as f:
                json.dump(visibility_data, f, indent=2)
            print(f"Visibility settings saved to {visibility_path}")
        except Exception as e:
            print(f"Failed to save visibility settings: {e}")
    
    def _loadLabelMetadata(self):
        """Load label metadata from a sidecar JSON file (including visibility settings)."""
        if self.tiff_mask_file:
            metadata_path = LabelMetadataStore.get_sidecar_path(self.tiff_mask_file)
            if self.labelMetadataStore.load(metadata_path):
                print(f"Label metadata loaded from {metadata_path}")
                self.uniqLabelList.set_metadata_store(self.labelMetadataStore)
                self._updateLabelStateStats()
                
                # Load visibility settings
                self._loadVisibilitySettings(metadata_path)
            else:
                # Initialize metadata for existing labels as MANUAL (legacy data)
                self._initializeMetadataForExistingLabels()
    
    def _loadVisibilitySettings(self, base_path: str):
        """Load visibility manager settings."""
        import json
        visibility_path = base_path.replace('.json', '_visibility.json')
        try:
            if os.path.exists(visibility_path):
                with open(visibility_path, 'r') as f:
                    visibility_data = json.load(f)
                self.visibilityManager.from_dict(visibility_data)
                
                # Sync UI with loaded settings
                self._syncVisibilityUIFromManager()
                print(f"Visibility settings loaded from {visibility_path}")
            else:
                # Use defaults for backward compatibility
                print("No visibility settings found, using defaults")
        except Exception as e:
            print(f"Failed to load visibility settings: {e}")
    
    def _syncVisibilityUIFromManager(self):
        """Sync UI elements with visibility manager state."""
        # Update list filter dropdown
        current_mode = self.visibilityManager.get_list_filter_mode()
        for i in range(self.listFilterCombo.count()):
            if self.listFilterCombo.itemData(i) == current_mode:
                self.listFilterCombo.setCurrentIndex(i)
                break
        
        # Update hide verified checkbox
        hide_verified = LabelState.VERIFIED in self.visibilityManager.get_view_hidden_states()
        self.hideVerifiedCheckbox.setChecked(hide_verified)
        
        # Update solo mode label
        if self.visibilityManager.is_solo_mode():
            solo_label = self.visibilityManager._solo_label
            self.soloModeLabel.setText(f"Solo: {solo_label}")
        else:
            self.soloModeLabel.setText("")
        
        # Apply list filter
        self.uniqLabelList.apply_state_filter(current_mode)
        
        # Sync checkbox states
        self.uniqLabelList.sync_checkbox_with_visibility_manager()
        
        # Update views
        self._updateAllViewVisibility()
    
    def _initializeMetadataForExistingLabels(self):
        """
        Initialize metadata for existing labels when loading a mask without metadata.
        Labels are marked as MANUAL origin with EDITED state (legacy data).
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return
        
        unique_labels = np.unique(self.tiffMask)
        for label in unique_labels:
            if label == 0:
                continue
            label_str = str(label)
            if label_str not in self.labelMetadataStore:
                self.labelMetadataStore.get_or_create(label_str, origin=LabelOrigin.MANUAL)
                # Mark as EDITED since we don't have the original proposed snapshot
                self.labelMetadataStore.mark_edited(label_str)
        
        self._updateLabelStateStats()
    
    # ============== End Label Lifecycle Management Methods ==============

    def split_label(self):
        """
        Split the target label into connected components via cc3d,
        and filter out components with a voxel count less than 100.
        """
        # 1) parse the target label from the input field
        try:
            target_label = int(self.label_input.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid integer label."
            )
            return
        size_threshold = 100  # Define size threshold

        # 2) ensure we have a 3D mask loaded
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No mask data available."
            )
            return

        # 3) Work only inside the target label's bounding box. The previous
        # implementation ran connected components on the full volume.
        mask = self.tiffMask
        bbox = self._label_bbox_3d(target_label)
        if bbox is None:
            QtWidgets.QMessageBox.information(
                self,
                "Label Not Found",
                f"Label {target_label} was not found in the mask."
            )
            return
        z1, z2, y1, y2, x1, x2 = bbox
        mask_roi = mask[z1:z2, y1:y2, x1:x2]
        target_roi = mask_roi == target_label
        target_voxel_count = int(np.count_nonzero(target_roi))

        # 4) label all connected components on the binary ROI
        #    returns 0..N where 0 is background, 1..N are components
        cc_map, num_components = cc3d.connected_components(
            target_roi,
            connectivity=26,
            return_N=True,
        )
        
        # [New] 4.5) Filter out connected components smaller than the threshold
        if num_components > 0:
            voxel_counts = np.bincount(cc_map.ravel())
            keep_components = np.flatnonzero(voxel_counts >= size_threshold)
            keep_components = keep_components[keep_components > 0]
        else:
            keep_components = np.array([], dtype=np.int64)
        num_components_after_filter = int(keep_components.size)

        if num_components_after_filter == 0:
            QtWidgets.QMessageBox.information(
                self,
                "No Components",
                f"No connected components larger than {size_threshold} voxels found for label {target_label}."
            )
            # Ensure original ROI region is cleared even if no components are found
            mask_roi[target_roi] = 0
            self._invalidate_shape_cache_for_bbox(bbox)
            self._markMaskDirty()
            self._updateCachedCountsForRelabel(target_label, target_voxel_count, [])
            self.openNextImg(nextN=0, store_history=False) # Refresh view
            return

        # 5) Keep the largest component as the original label. Smaller
        # components receive new label IDs.
        offset = int(mask.max())
        largest_component = int(keep_components[np.argmax(voxel_counts[keep_components])])
        split_components = keep_components[keep_components != largest_component]
        new_labels = np.arange(
            offset + 1,
            offset + split_components.size + 1,
            dtype=mask.dtype,
        )
        component_to_label = np.zeros(int(cc_map.max()) + 1, dtype=mask.dtype)
        component_to_label[largest_component] = target_label
        component_to_label[split_components] = new_labels
        
        relabeled_roi = component_to_label[cc_map]
        mask_roi[target_roi] = 0
        relabeled_pixels = relabeled_roi > 0
        mask_roi[relabeled_pixels] = relabeled_roi[relabeled_pixels]

        # 6) update the in‐memory mask and enable saving
        self._invalidate_shape_cache_for_bbox(bbox)
        self._markMaskDirty()
        new_label_counts = [(target_label, int(voxel_counts[largest_component]))]
        new_label_counts.extend(zip(new_labels, voxel_counts[split_components]))
        self._updateCachedCountsForRelabel(
            target_label,
            target_voxel_count,
            new_label_counts,
        )

        # 6.5) Update metadata for split labels. If only one component remains,
        # keep it as the original label instead of creating self-parent metadata.
        if new_labels.size > 0:
            child_labels = [str(target_label)] + [str(label) for label in new_labels]
            self.labelMetadataStore.handle_split(
                parent_label=str(target_label),
                child_labels=child_labels,
                child_masks={},
                push_undo=True
            )
        else:
            self.labelMetadataStore.set_state(str(target_label), LabelState.EDITED, push_undo=True)
        self._updateLabelStateStats()

        # 7) refresh the displayed slice immediately
        self.openNextImg(nextN=0, store_history=False)

        # 8) [Change] Inform the user how many components were created after filtering
        QtWidgets.QMessageBox.information(
            self,
            "Split Completed",
            f"Label {target_label} kept the largest component. "
            f"Created {new_labels.size} new label(s) from smaller components "
            f"(size >= {size_threshold})."
        )

    def clear_watershed_seeds(self):
        """Clear all 3D watershed seed points"""
        self.canvas.clearWatershedSeeds()
        self.watershed_3d_label_input.clear()  # Clear displayed label
        self.statusBar().showMessage("Cleared all watershed seed points.")
        QTimer.singleShot(2000, lambda: self.statusBar().clearMessage())

    def handleWatershedSeedClick(self, x, y, slice_idx):
        """Handle 3D watershed seed point click event"""
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("Please load a mask file first.")
            return
        
        # Convert 2D canvas coordinates to 3D coordinates based on current view axis
        if self.currentViewAxis == 0:  # Axial view (XY plane, Z varies)
            x_3d, y_3d, z_3d = int(x), int(y), int(slice_idx)
        elif self.currentViewAxis == 1:  # Coronal view (XZ plane, Y varies)
            x_3d, y_3d, z_3d = int(x), int(slice_idx), int(y)
        elif self.currentViewAxis == 2:  # Sagittal view (YZ plane, X varies)
            x_3d, y_3d, z_3d = int(slice_idx), int(x), int(y)
        else:
            self.statusBar().showMessage("Invalid view axis.")
            return
        
        # Validate 3D coordinates are within bounds
        if not (0 <= z_3d < self.tiffMask.shape[0] and 
                0 <= y_3d < self.tiffMask.shape[1] and 
                0 <= x_3d < self.tiffMask.shape[2]):
            self.statusBar().showMessage(f"Click position out of bounds: 3D({x_3d}, {y_3d}, {z_3d}), shape{self.tiffMask.shape}")
            return
        
        # Get the label value at the clicked position using 3D coordinates
        clicked_label = int(self.tiffMask[z_3d, y_3d, x_3d])
        
        if clicked_label == 0:
            self.statusBar().showMessage("Please click on a labeled region (not background).")
            return
        
        # Check if this is the first seed point
        if not self.canvas.watershed_seed_points:
            # First seed point: set target label
            self.canvas.watershed_auto_label = clicked_label
            self.watershed_3d_label_input.setText(str(clicked_label))
            
            # Add seed point with 3D coordinates
            seed_point = {
                'x_3d': x_3d,
                'y_3d': y_3d,
                'z_3d': z_3d,
                'label': clicked_label,
                'view_axis': self.currentViewAxis  # Store which axis it was placed in
            }
            self.canvas.watershed_seed_points.append(seed_point)
            self.canvas.update()
            
            self.statusBar().showMessage(f"Added first seed point for label {clicked_label} at 3D coords ({x_3d}, {y_3d}, {z_3d}).")
            
        else:
            # Check whether the new seed point is on the same label
            if clicked_label != self.canvas.watershed_auto_label:
                self.statusBar().showMessage(
                    f"Error: Clicked on label {clicked_label}, but previous seeds are on label {self.canvas.watershed_auto_label}. "
                    f"Please click 'Clear Seeds' and start over, or click on label {self.canvas.watershed_auto_label}."
                )
                return
            
            # Add seed point with 3D coordinates
            seed_point = {
                'x_3d': x_3d,
                'y_3d': y_3d,
                'z_3d': z_3d,
                'label': clicked_label,
                'view_axis': self.currentViewAxis
            }
            self.canvas.watershed_seed_points.append(seed_point)
            self.canvas.update()
            
            self.statusBar().showMessage(f"Added seed point #{len(self.canvas.watershed_seed_points)} for label {clicked_label} at 3D coords ({x_3d}, {y_3d}, {z_3d}).")
        
        QTimer.singleShot(3000, lambda: self.statusBar().clearMessage())

    def apply_3d_watershed(self):
        """Perform optimized 3D watershed segmentation - accelerate using bounding box restriction"""
        # Use the auto-detected label
        target_label = self.canvas.getWatershedAutoLabel()
        if target_label is None:
            self.statusBar().showMessage("Please place seed points first by clicking in watershed_3d mode.")
            return

        seed_points = self.canvas.getWatershedSeeds()
        if not seed_points:
            self.statusBar().showMessage("Please place seed points first by clicking in watershed_3d mode.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("Mask data not available for 3D watershed.")
            return

        self.statusBar().showMessage(f"Applying optimized 3D watershed to label {target_label} with {len(seed_points)} seed points...")

        try:
            bbox = self._label_bbox_3d(target_label, padding=5)
            if bbox is None:
                self.statusBar().showMessage(f"Label {target_label} not found in the mask.")
                return
            z_min, z_max, y_min, y_max, x_min, x_max = bbox

            # Store only the affected region for undo instead of copying the
            # whole volume.
            undo_region = self.tiffMask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            self._watershed_undo_stack.append(("region", bbox, undo_region))
            if len(self._watershed_undo_stack) > WATERSHED_UNDO_LIMIT:
                self._watershed_undo_stack.pop(0)
            self._watershed_redo_stack.clear()

            # Display bounding box info
            subvolume_size = f"{z_max-z_min}x{y_max-y_min}x{x_max-x_min}"
            original_size = f"{self.tiffMask.shape[0]}x{self.tiffMask.shape[1]}x{self.tiffMask.shape[2]}"
            self.statusBar().showMessage(f"Processing subvolume {subvolume_size} from original {original_size}...")
            
            # Extract subregion
            mask_subregion = self.tiffMask[z_min:z_max, y_min:y_max, x_min:x_max]
            target_subregion = mask_subregion == target_label
            target_voxel_count = int(np.count_nonzero(target_subregion))
            
            # Create seed point markers within the subregion
            markers_sub = np.zeros_like(target_subregion, dtype=np.int32)
            for i, seed in enumerate(seed_points):
                # Use 3D coordinates from seed
                x_3d, y_3d, z_3d = seed['x_3d'], seed['y_3d'], seed['z_3d']
                # Convert to subregion coordinates
                z_sub = z_3d - z_min
                y_sub = y_3d - y_min
                x_sub = x_3d - x_min
                if (0 <= z_sub < target_subregion.shape[0] and 
                    0 <= y_sub < target_subregion.shape[1] and 
                    0 <= x_sub < target_subregion.shape[2]):
                    markers_sub[z_sub, y_sub, x_sub] = i + 1
            
            # 🚀 Run watershed on the subregion with iterative filtering for small regions
            distance_sub = ndi.distance_transform_edt(target_subregion)
            from skimage.segmentation import watershed
            
            # Iterative watershed with small region filtering
            MIN_REGION_SIZE = 50  # Minimum region size in voxels
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                ws_labels_sub = watershed(-distance_sub, markers_sub, mask=target_subregion)
                
                # Check region sizes
                region_sizes = np.bincount(ws_labels_sub.ravel())
                unique_regions_sub = np.flatnonzero(region_sizes)
                unique_regions_sub = unique_regions_sub[unique_regions_sub > 0]
                small_regions = unique_regions_sub[region_sizes[unique_regions_sub] < MIN_REGION_SIZE]
                
                if small_regions.size == 0:
                    # All regions are large enough, we're done
                    break
                
                # Remove markers for small regions and re-run watershed
                self.statusBar().showMessage(
                    f"Iteration {iteration+1}: Removing {small_regions.size} small regions (size < {MIN_REGION_SIZE})..."
                )
                
                # Remove markers corresponding to small regions
                markers_sub[np.isin(markers_sub, small_regions)] = 0
                
                iteration += 1
            
            # Update mask - replace original target_label region with watershed result
            self._invalidate_shape_cache_for_bbox(bbox)
            max_existing_label = int(self.tiffMask.max())
            region_sizes = np.bincount(ws_labels_sub.ravel())
            unique_regions = np.flatnonzero(region_sizes)
            unique_regions = unique_regions[unique_regions > 0]
            if unique_regions.size == 0:
                if (
                    self._watershed_undo_stack
                    and isinstance(self._watershed_undo_stack[-1], tuple)
                    and self._watershed_undo_stack[-1][0] == "region"
                    and self._watershed_undo_stack[-1][1] == bbox
                ):
                    self._watershed_undo_stack.pop()
                self.statusBar().showMessage(
                    f"Watershed produced no regions for label {target_label}; label was left unchanged."
                )
                QTimer.singleShot(3000, lambda: self.statusBar().clearMessage())
                return
            largest_region = int(unique_regions[np.argmax(region_sizes[unique_regions])])
            split_regions = unique_regions[unique_regions != largest_region]
            new_labels_array = np.arange(
                max_existing_label + 1,
                max_existing_label + split_regions.size + 1,
                dtype=self.tiffMask.dtype,
            )
            region_to_label = np.zeros(int(ws_labels_sub.max()) + 1, dtype=self.tiffMask.dtype)
            region_to_label[largest_region] = target_label
            region_to_label[split_regions] = new_labels_array
            relabeled_sub = region_to_label[ws_labels_sub]

            mask_subregion[target_subregion] = 0
            relabeled_pixels = relabeled_sub > 0
            mask_subregion[relabeled_pixels] = relabeled_sub[relabeled_pixels]

            # Refresh UI
            self._markMaskDirty()
            new_label_counts = [(target_label, int(region_sizes[largest_region]))]
            new_label_counts.extend(zip(new_labels_array, region_sizes[split_regions]))
            self._updateCachedCountsForRelabel(
                target_label,
                target_voxel_count,
                new_label_counts,
            )
            self.openNextImg(nextN=0, store_history=False)  # Refresh current slice display
            
            # Register new labels with PROPOSED state (from watershed)
            new_labels = [int(label) for label in new_labels_array]
            if new_labels:
                self._registerAutoSegmentationLabels(
                    new_labels,
                    LabelOrigin.WATERSHED,
                    store_snapshots=False,
                )
            
            # Clear seed points
            self.canvas.clearWatershedSeeds()
            
            # Show optimization effect information
            volume_reduction = ((z_max-z_min) * (y_max-y_min) * (x_max-x_min)) / (self.tiffMask.shape[0] * self.tiffMask.shape[1] * self.tiffMask.shape[2])
            speedup_estimate = 1 / volume_reduction if volume_reduction > 0 else 1
            
            iteration_info = f" (filtered in {iteration} iteration{'s' if iteration != 1 else ''})" if iteration > 0 else ""
            self.statusBar().showMessage(
                f"🚀 Optimized 3D watershed completed! "
                f"Kept label {target_label} for the largest region; "
                f"created {len(new_labels)} new label(s){iteration_info}. "
                f"Subvolume: {subvolume_size} "
                f"Speedup: ~{speedup_estimate:.1f}x"
            )
            QTimer.singleShot(5000, lambda: self.statusBar().clearMessage())

        except Exception as e:
            self.statusBar().showMessage(f"Error in optimized 3D watershed: {str(e)}")
            QTimer.singleShot(3000, lambda: self.statusBar().clearMessage())

    def compute_bbox_3d(self, binary_mask):
        """
        Compute the bounding box of a 3D binary mask.
        
        Args:
            binary_mask (numpy.ndarray): 3D binary mask
            
        Returns:
            tuple: (z_min, z_max, y_min, y_max, x_min, x_max) or None
        """
        if not np.any(binary_mask):
            return None
            
        # Find coordinates of all non-zero voxels
        coords = np.where(binary_mask)
        
        if len(coords[0]) == 0:
            return None
            
        # Compute min and max for each dimension
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        return z_min, z_max, y_min, y_max, x_min, x_max

    def count_large_components(self, binary_mask, min_size=10):
        """
        Counts the number of connected components larger than a minimum size.
        """
        if not np.any(binary_mask):
            return 0

        labels_out = cc3d.connected_components(binary_mask, connectivity=8)
        if labels_out.max() == 0:
            return 0

        stats = cc3d.statistics(labels_out)
        voxel_counts = stats['voxel_counts'][1:]
        
        num_large_components = np.sum(voxel_counts >= min_size)
        
        return num_large_components

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))
            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty or not self.actions.saveMask.isEnabled():
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            self._cleanupTempMaskAutosave()
            self._mask_autosave_dirty = False
            self._last_autosave_revision = getattr(self, "_mask_edit_revision", 0)
            return True
        elif answer == mb.Save:
            self.saveMask()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]



    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    @property
    def imageList(self):
        """Return empty list since file list is removed."""
        return []

    def importDroppedImageFiles(self, imageFiles):
        """Load the first dropped image file."""
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        # Add TIFF and NIfTI extensions
        extensions.extend(['.tif', '.tiff', '.nii', '.nii.gz'])

        for file in imageFiles:
            if file.lower().endswith(tuple(extensions)):
                self.loadFile(file)
                break

    def importDirImages(self, dirpath, pattern=None, load=True):
        """Import directory - simplified version without file list."""
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None

        # Scan for images and load the first one if load=True
        if load:
            filenames = self.scanAllImages(dirpath)
            if pattern:
                try:
                    filenames = [f for f in filenames if re.search(pattern, f)]
                except re.error:
                    pass
            if filenames:
                self.loadFile(filenames[0])

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    # Add this condition to filter out _mask.tiff files
                    if not relativePath.lower().endswith('_mask.tiff'):
                        images.append(relativePath)
        
        images = natsort.os_sorted(images)
        print(f"Found {len(images)} images in {folderPath}")
        return images

    def show_interpolate_dialog(self):
        """
        Show the interpolation dialog and, based on the most recently used label,
        intelligently compute the largest discontinuity as the default start/end slices.
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available to interpolate.")
            return

        # --- Begin new computation logic ---
        
        # 1. Use the most recently operated label as the default target
        target_label = int(self.recent_label)
        
        # 2. Find all slice indices where the label exists
        positions = np.argwhere(self.tiffMask == target_label)
        
        start_slice, end_slice = 0, 0
        
        if positions.size > 0:
            # For the current view, get all unique slice indices containing the label and sort
            slice_indices_for_view = np.unique(positions[:, self.currentViewAxis])
            
            # 3. If fewer than 2 slices contain the label, use default values
            if len(slice_indices_for_view) < 2:
                start_slice = self.currentSliceIndex
                end_slice = self.currentSliceIndex + 10
            else:
                # 4. Compute gaps between consecutive slices
                gaps = np.diff(slice_indices_for_view)
                
                if gaps.size > 0:
                    # 5. Find the index of the largest gap
                    largest_gap_index = np.argmax(gaps)
                    # The start slice is before the largest gap
                    start_slice = int(slice_indices_for_view[largest_gap_index])
                    # The end slice is after the largest gap
                    end_slice = int(slice_indices_for_view[largest_gap_index + 1])
                else: # If there is only one gap
                    start_slice = int(slice_indices_for_view[0])
                    end_slice = int(slice_indices_for_view[1])
        else:
            # If the label does not exist in the mask, use default values
            start_slice = self.currentSliceIndex
            end_slice = self.currentSliceIndex + 10
            
        # 6. Determine the maximum slice value for the dialog sliders
        max_slice_for_view = self.tiffData.shape[self.currentViewAxis] - 1
        
        # 7. Create and show the dialog, prefilled with the computed values
        dialog = InterpolateDialog(self, start_slice, end_slice, max_slice_for_view)
        dialog.target_label_input.setText(str(target_label)) # Prefill with recent label

        # --- End new logic ---

        if dialog.exec_():
            s_slice, e_slice, label_str = dialog.getValues()
            
            if not label_str.isdigit():
                QtWidgets.QMessageBox.critical(self, "Error", "Target Label must be an integer.")
                return

            label_to_interpolate = int(label_str)
            
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.run_interpolation(s_slice, e_slice, label_to_interpolate)
                
                # If we just interpolated the boundary label, remove it from the mask afterwards
                if label_to_interpolate == 10000:
                    self.tiffMask[self.tiffMask == 10000] = 0
                    
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Interpolation Error", str(e))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

    def run_interpolation(self, start_slice, end_slice, target_label):
        """Perform interpolation based on signed distance transform."""
        if start_slice >= end_slice:
            raise ValueError("Start Slice must be smaller than End Slice.")

        # 1. Get masks for start and end slices
        mask_a = (self.get_current_slice(self.tiffMask, start_slice) == target_label)
        mask_b = (self.get_current_slice(self.tiffMask, end_slice) == target_label)

        if not mask_a.any() or not mask_b.any():
            raise ValueError(f"Label {target_label} not found on both start and end slices.")

        # 2. Compute signed distance fields (inside positive, outside negative)
        dt_a = distance_transform_edt(mask_a) - distance_transform_edt(~mask_a)
        dt_b = distance_transform_edt(mask_b) - distance_transform_edt(~mask_b)
        
        # 3. Iterate through intermediate slices and interpolate
        total_slices = end_slice - start_slice
        for i in range(1, total_slices):
            slice_index = start_slice + i
            
            # Compute interpolation weight for current slice
            weight = i / total_slices
            
            # Linearly interpolate distance fields
            interp_dt = (1.0 - weight) * dt_a + weight * dt_b
            
            # Reconstruct mask from interpolated distance field (distance >= 0 is inside)
            interp_mask = interp_dt >= 0
            
            # 4. Write the generated mask back into self.tiffMask
            current_slice_mask = self.get_current_slice(self.tiffMask, slice_index)
            # Clear any old labels in this area first, then fill with new label
            current_slice_mask[interp_mask] = target_label
            # If needed, keep other labels:
            # current_slice_mask[~interp_mask & (current_slice_mask == target_label)] = 0

        # 5. Refresh UI
        self._invalidate_shape_cache()
        self._markMaskDirty()
        self.updateUniqueLabelListFromEntireMask()
        self.openNextImg(nextN=0, store_history=False)  # Refresh current view
        self.status("Interpolation completed successfully.") 
        # QtWidgets.QMessageBox.information(
        #     self, "Success", f"Successfully interpolated label {target_label} between slices {start_slice} and {end_slice}."
        # )


    def onUniqLabelVisibilityChanged(self, label: str, visible: bool):
        """Batch update label visibility; reentrancy-safe and optionally block uniqLabelList signals."""
        # Reentrancy guard: if already handling, just return (or update state as needed)
        if getattr(self, "_handling_visibility", False):
            return
        self._handling_visibility = True
        try:
            # 1) Record state
            self.label_visibility_states[label] = visible

            # 2) Batch set visibility for existing shapes on the canvas (single redraw)
            shapes = [s for s in self.canvas.shapes if s.label == label]
            if shapes:
                # Assume canvas.setShapesVisible accepts a dict and triggers only one update()
                self.canvas.setShapesVisible({s: visible for s in shapes})

            # 3) If toggled visible but no shape on current slice, create on demand
            if visible and not shapes and self.tiffMask is not None:
                mask2d = self.get_current_slice(self.tiffMask, self.currentSliceIndex)
                lab = int(label)
                if (mask2d == lab).any():
                    y1, y2, x1, x2, roi_mask = self._fast_bbox_and_roi(mask2d, lab)
                    shape = Shape(
                        label=str(label),
                        shape_type="mask",
                        description=f"Mask for label {label}",
                        slice_id=self.currentSliceIndex,
                    )
                    shape.setShapeRefined(
                        shape_type="mask",
                        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                        point_labels=[1, 1],
                        mask=roi_mask,
                    )

                    # Temporarily block uniqLabelList signals to avoid callbacks triggered by addLabelMinimal
                    blocker_used = False
                    if hasattr(self, "uniqLabelList") and hasattr(self.uniqLabelList, "blockSignals"):
                        self.uniqLabelList.blockSignals(True)
                        blocker_used = True
                    try:
                        # Add the label (try to avoid triggering visibility callbacks in addLabelMinimal,
                        # or control signal emission inside it)
                        self.addLabelMinimal(shape)
                    finally:
                        if blocker_used:
                            self.uniqLabelList.blockSignals(False)

                    # Add the shape to the canvas (loadShapes typically does not trigger the same signal)
                    self.canvas.loadShapes([shape], replace=False)
            
            # 4) 3D view/other sync (should not trigger back into onUniqLabelVisibilityChanged)
            try:
                lbl_int = int(label)
                if hasattr(self, "vtk_widget") and self.vtk_widget:
                    self.vtk_widget.toggle_label_visibility(lbl_int, visible)
            except Exception:
                pass

        finally:
            self._handling_visibility = False
    
    # ---- State-Based Visibility Management Handlers ----
    
    def _onListFilterChanged(self, index: int):
        """Handle list filter dropdown change."""
        filter_mode = self.listFilterCombo.itemData(index)
        if filter_mode is None:
            filter_mode = LabelFilterMode.ALL
        self.visibilityManager.set_list_filter_mode(filter_mode)
        self.uniqLabelList.apply_state_filter(filter_mode)
        self._updateLabelCounter()
        # Update status bar with filter info
        visible_count = self.uniqLabelList.get_visible_item_count()
        total_count = self.uniqLabelList.count()
        if filter_mode != LabelFilterMode.ALL:
            self.status(f"Showing {visible_count}/{total_count} labels ({filter_mode.value})")
    
    def _onHideVerifiedChanged(self, state: int):
        """Handle hide-verified checkbox change."""
        hide_verified = (state == Qt.Checked)
        self.visibilityManager.set_hide_verified_in_views(hide_verified)
        
        # Update views
        self._updateAllViewVisibility()
        
        # Check if selected label became hidden
        selected_items = self.uniqLabelList.selectedItems()
        if selected_items:
            label = selected_items[0].data(Qt.UserRole)
            if not self.visibilityManager.get_effective_visible(label):
                self.status(f"Selected label {label} is hidden in views.")
    
    def _onSoloCurrentFromButton(self):
        """Handle Solo button click - solo the selected label."""
        selected_items = self.uniqLabelList.selectedItems()
        if selected_items:
            label = selected_items[0].data(Qt.UserRole)
            self._onSoloCurrentRequested(label)
        else:
            self.status("No label selected for solo mode.")
    
    def _onSoloCurrentRequested(self, label: str):
        """Handle solo current request from context menu or signal."""
        self.visibilityManager.set_solo_mode(label)
        self.soloModeLabel.setText(f"Solo: {label}")
        self._updateAllViewVisibility()
        self.status(f"Solo mode: showing only label {label}")
    
    def _onShowAllRequested(self):
        """Handle show all request - exit solo mode and show all labels."""
        self.visibilityManager.show_all()  # Already clears solo mode; avoids duplicate signal cascade
        self.soloModeLabel.setText("")
        self.uniqLabelList.sync_checkbox_with_visibility_manager()
        self._updateAllViewVisibility()
        self.status("Showing all labels")
    
    def _onLabelDoubleClicked(self, label: str):
        """Handle double-click on label to jump to its middle slice."""
        if getattr(self, "_labelJumpInProgress", False):
            return
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.status("No 3D mask loaded.")
            return

        self._labelJumpInProgress = True
        try:
            label_int = int(label)
        except (ValueError, TypeError):
            self._labelJumpInProgress = False
            self.status(f"Invalid label: {label}")
            return

        # Find all slices that contain this label along the current view axis
        axis = self.currentViewAxis
        slices_with_label = []
        
        # Get the number of slices along the current axis
        num_slices = self.tiffMask.shape[axis]
        
        # Efficiently find slices containing the label
        for slice_idx in range(num_slices):
            # Get the slice based on the current view axis
            if axis == 0:  # Axial (Z)
                slice_data = self.tiffMask[slice_idx, :, :]
            elif axis == 1:  # Coronal (Y)
                slice_data = self.tiffMask[:, slice_idx, :]
            else:  # Sagittal (X)
                slice_data = self.tiffMask[:, :, slice_idx]
            
            if np.any(slice_data == label_int):
                slices_with_label.append(slice_idx)
        
        if not slices_with_label:
            self._labelJumpInProgress = False
            self.status(f"Label {label} not found in any slice.")
            return

        # Calculate the middle slice
        middle_idx = len(slices_with_label) // 2
        target_slice = slices_with_label[middle_idx]
        
        # Navigate to the target slice
        slice_diff = target_slice - self.currentSliceIndex
        if slice_diff > 0:
            self.openNextImg(nextN=slice_diff)
        elif slice_diff < 0:
            self.openPrevImg(nextN=-slice_diff)
        # Fallback: clear jump-in-progress if loadAnnotationsAndMasks never runs (e.g. navigation edge case)
        QTimer.singleShot(2000, lambda: setattr(self, "_labelJumpInProgress", False))

        self.status(f"Jumped to slice {target_slice} (middle of label {label}, spans slices {slices_with_label[0]}-{slices_with_label[-1]})")
    
    # ---- Label Search Methods ----
    
    def _onLabelSearchChanged(self, text: str):
        """Handle label search text change - filter the list."""
        text = text.strip()
        
        for row in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(row)
            if item:
                label = item.data(Qt.UserRole)
                # Show item if search is empty or label contains the search text
                if not text:
                    # Respect state filter when search is cleared
                    filter_mode = self.listFilterCombo.itemData(self.listFilterCombo.currentIndex())
                    if filter_mode is None:
                        filter_mode = LabelFilterMode.ALL
                    state = None
                    if hasattr(self, 'labelMetadataStore'):
                        state = self.labelMetadataStore.get_state(label)
                    passes_filter = filter_mode.matches_state(state) if filter_mode != LabelFilterMode.ALL else True
                    item.setHidden(not passes_filter)
                else:
                    # Search by label ID (partial match)
                    matches = text.lower() in str(label).lower()
                    item.setHidden(not matches)
        
        # Auto-select and scroll to exact match
        if text:
            exact_item = self.uniqLabelList.findItemByLabel(text)
            if exact_item:
                self.uniqLabelList.setCurrentItem(exact_item)
                self.uniqLabelList.scrollToItem(exact_item)
    
    def _onLabelSearchEnter(self):
        """Handle Enter key in search box - jump to the label's middle slice."""
        # Debounce: ignore rapid repeat Enter to prevent crash from overlapping work
        if getattr(self, "_labelSearchEnterCooldown", False):
            return
        if getattr(self, "_labelJumpInProgress", False):
            return
        self._labelSearchEnterCooldown = True
        QTimer.singleShot(800, lambda: setattr(self, "_labelSearchEnterCooldown", False))

        text = self.labelSearchBox.text().strip()
        if not text:
            return

        # Try to find the exact label
        item = self.uniqLabelList.findItemByLabel(text)
        if item:
            # Select the item
            self.uniqLabelList.setCurrentItem(item)
            self.uniqLabelList.scrollToItem(item)
            # Jump to its middle slice
            self._onLabelDoubleClicked(text)
        else:
            # Try partial match - jump to first visible item
            for row in range(self.uniqLabelList.count()):
                item = self.uniqLabelList.item(row)
                if item and not item.isHidden():
                    label = item.data(Qt.UserRole)
                    self.uniqLabelList.setCurrentItem(item)
                    self._onLabelDoubleClicked(label)
                    break
            else:
                self.status(f"Label '{text}' not found.")
    
    def _clearLabelSearch(self):
        """Clear the label search box."""
        self.labelSearchBox.clear()
        # Re-apply state filter
        filter_mode = self.listFilterCombo.itemData(self.listFilterCombo.currentIndex())
        if filter_mode is None:
            filter_mode = LabelFilterMode.ALL
        self.uniqLabelList.apply_state_filter(filter_mode)
    
    def _onSoloModeChanged(self, is_solo: bool, label: str):
        """Handle solo mode changes. Reload slice with solo-filtered shapes for faster display."""
        if is_solo and label:
            self.soloModeLabel.setText(f"Solo: {label}")
        else:
            self.soloModeLabel.setText("")
        # Update 3D overlay with solo label
        self._update_3d_cache_overlay()
        # Reload current slice so canvas gets only solo label's shapes (faster painting)
        if hasattr(self, "tiffData") and self.tiffData is not None and hasattr(self, "tiffMask") and self.tiffMask is not None:
            self.loadAnnotationsAndMasks()
        else:
            self._updateAllViewVisibility()
    
    def _onAllVisibilityChanged(self):
        """Handle bulk visibility changes."""
        self._updateAllViewVisibility()
    
    def _onEffectiveVisibilityChanged(self, label: str, visible: bool):
        """Handle effective visibility change for a single label."""
        # Update 2D view
        shapes = [s for s in self.canvas.shapes if s.label == label]
        if shapes:
            self.canvas.setShapesVisible({s: visible for s in shapes})
        
        # Update 3D view
        try:
            lbl_int = int(label)
            if hasattr(self, "vtk_widget") and self.vtk_widget:
                self.vtk_widget.toggle_label_visibility(lbl_int, visible)
        except Exception:
            pass
    
    def _updateAllViewVisibility(self):
        """Update visibility for all labels in both 2D and 3D views."""
        # Get all hidden labels
        hidden_labels = self.visibilityManager.get_hidden_label_ids()
        
        # Update 2D shapes (batch update with single redraw)
        visibility_map = {}
        for shape in self.canvas.shapes:
            label = shape.label
            if label:
                visible = label not in hidden_labels
                visibility_map[shape] = visible
        
        if visibility_map:
            self.canvas.setShapesVisible(visibility_map)
        
        # Update 3D view (batch update with single render)
        if hasattr(self, "vtk_widget") and self.vtk_widget:
            vtk_visibility = {}
            for row in range(self.uniqLabelList.count()):
                item = self.uniqLabelList.item(row)
                if item:
                    label = item.data(Qt.UserRole)
                    try:
                        lbl_int = int(label)
                        visible = label not in hidden_labels
                        vtk_visibility[lbl_int] = visible
                    except (ValueError, TypeError):
                        pass
            
            # Use batch method for much better performance
            if vtk_visibility:
                self.vtk_widget.set_labels_visibility_batch(vtk_visibility)
        
        # Force immediate 2D repaint (update() is async and can lag when event loop is busy)
        self.canvas.repaint()
    
    def _onShortcutsDockVisibilityChanged(self, visible):
        """
        Track floating state and fix appearance when the shortcuts dock becomes visible.
        - Re-float when reopened after close (macOS red button fix)
        - Bring to front and ensure on-screen (single deferred call to avoid flashing)
        """
        dock = self.shortcuts_dock
        if visible:
            self._shortcuts_dock_was_floating = True
            # Defer all appearance fixes to next event loop so Qt finishes its show
            # sequence first. Running raise/activate/move immediately causes flashing.
            QtCore.QTimer.singleShot(50, self._deferredShortcutsDockAppearance)
        else:
            self._shortcuts_dock_was_floating = dock.isFloating()

    def _deferredShortcutsDockAppearance(self):
        """Run once after dock becomes visible: re-float if needed, bring to front, clamp on-screen."""
        dock = self.shortcuts_dock
        if not dock.isVisible():
            return
        # Re-float when reopened (macOS red button fix). Only if was floating and isn't now.
        if getattr(self, "_shortcuts_dock_was_floating", True) and not dock.isFloating():
            dock.setFloating(True)
        if not dock.isFloating():
            return
        dock.raise_()
        dock.activateWindow()
        # Clamp to screen
        try:
            app = QtWidgets.QApplication.instance()
            if not (app and hasattr(app, "primaryScreen") and app.primaryScreen()):
                return
            avail = app.primaryScreen().availableGeometry()
            win = dock.window()
            if win and win != self:
                x, y = win.x(), win.y()
                w, h = win.width(), win.height()
            else:
                pos = dock.mapToGlobal(QtCore.QPoint(0, 0))
                x, y = pos.x(), pos.y()
                w, h = dock.width(), dock.height()
            dx = min(0, avail.right() - (x + w))
            dx = max(dx, avail.left() - x)
            dy = min(0, avail.bottom() - (y + h))
            dy = max(dy, avail.top() - y)
            if dx != 0 or dy != 0:
                target = dock.window() if dock.window() != self else dock
                if target:
                    target.move(x + dx, y + dy)
        except Exception:
            pass
    
    # ============== Keyboard Shortcut System ==============
    
    def _shortcutAllowed(self) -> bool:
        """
        Check if shortcuts should be processed.
        Always returns True: letter keys in textboxes should trigger shortcuts (e.g. F=verify, R=revert).
        """
        return True
    
    def _sc(self, key, default=None):
        """Get shortcut value from config; supports str or list."""
        val = self._config["shortcuts"].get(key, default)
        if val is None:
            return default
        return val

    def _sc_keys(self, key, default=None):
        """Get shortcut key(s) as list for setShortcuts/setShortcut."""
        val = self._sc(key, default)
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return val
        return [val]

    def _installShortcuts(self):
        """
        Install keyboard shortcuts from config.
        Called at the end of __init__ after all widgets/actions are created.
        """
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence

        sc = self._config["shortcuts"]

        # ---- Mode switch shortcuts (assign to existing actions) ----
        for action_key, action_obj in [
            ("select_mode", self.actions.selectMode),
            ("create_brush_mode", self.actions.createBrushMode),
            ("create_ai_mask_mode", self.actions.createAiMaskMode),
            ("create_rectangle_mode", self.actions.createBoxAiMaskMode),
            ("create_ai_boundary_mode", self.actions.createAiBoundaryMode),
            ("erase_mode", self.actions.createBoxEraseMode),
            ("create_watershed_3d_mode", self.actions.createWatershed3dMode),
        ]:
            keys = self._sc_keys(action_key)
            if keys:
                action_obj.setShortcuts(keys) if len(keys) > 1 else action_obj.setShortcut(keys[0])

        # Helper to create shortcuts with proper context
        def make_shortcut(key_val, handler):
            if key_val is None:
                return None
            ks = QKeySequence(key_val) if not isinstance(key_val, (list, tuple)) else QKeySequence(key_val[0])
            if ks.isEmpty():
                return None
            shortcut = QShortcut(ks, self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(handler)
            return shortcut

        # View axis shortcuts disabled (use Axis dropdown to switch views)

        # Label workflow shortcuts
        self._sc_verify = make_shortcut(sc.get("verify_label", "F"), self._shortcut_verifyLabel)
        self._sc_revert = make_shortcut(sc.get("revert_label", "R"), self._shortcut_revertLabel)
        reject_keys = sc.get("reject_label", ["Delete", "Backspace"])
        reject_list = reject_keys if isinstance(reject_keys, (list, tuple)) else [reject_keys]
        self._sc_delete = make_shortcut(reject_list[0] if reject_list else "Delete", self._shortcut_rejectLabel)
        self._sc_backspace = make_shortcut(reject_list[1] if len(reject_list) > 1 else "Backspace", self._shortcut_rejectLabel)

        # Commit (Ctrl+Return, Meta+Return)
        commit_keys = sc.get("commit", ["Ctrl+Return", "Meta+Return"])
        commit_list = commit_keys if isinstance(commit_keys, (list, tuple)) else [commit_keys]
        self._sc_commit = make_shortcut(commit_list[0] if commit_list else "Ctrl+Return", self._shortcut_commit)
        self._sc_commit_mac = make_shortcut(commit_list[1] if len(commit_list) > 1 else "Meta+Return", self._shortcut_commit)

        # Hide verified, Solo, Show all
        self._sc_hide = make_shortcut(sc.get("hide_verified", "H"), self._shortcut_toggleHideVerified)
        self._sc_solo = make_shortcut(sc.get("solo_label", "S"), self._shortcut_soloLabel)
        self._sc_show_all = make_shortcut(sc.get("show_all_labels", "Shift+S"), self._shortcut_showAll)

        # Search/Focus
        self._sc_search = make_shortcut(sc.get("focus_label_search", "Ctrl+F"), self._shortcut_focusLabelSearch)
        self._sc_brush_label = make_shortcut(sc.get("focus_brush_label", "Ctrl+L"), self._shortcut_focusBrushLabel)

        # Brush label
        self._sc_set_brush_label_0 = make_shortcut(sc.get("set_brush_label_0", "0"), self._shortcut_setBrushLabel0)

        # 3D
        self._sc_3d = make_shortcut(sc.get("toggle_3d", "Ctrl+3"), self._shortcut_toggle3D)

        # Escape
        self._sc_escape = make_shortcut(sc.get("escape", "Escape"), self._shortcut_escape)
    
    # ---- Shortcut handler methods ----

    def _reloadShortcuts(self):
        """Reload config and re-apply shortcuts after user saves from dialog."""
        config_path = self._config_file
        if config_path is None or not isinstance(config_path, str):
            config_path = get_user_config_path()
        config_path = osp.expanduser(config_path)
        if not osp.isfile(config_path):
            return
        self._config = get_config(config_path)
        sc = self._config["shortcuts"]
        if hasattr(self, "shortcuts_widget"):
            self.shortcuts_widget.set_shortcuts(sc)
        from qtpy.QtGui import QKeySequence

        def _apply_action_shortcut(action_obj, action_key):
            if action_obj is None:
                return
            keys = self._sc_keys(action_key)
            if keys:
                action_obj.setShortcuts(keys) if len(keys) > 1 else action_obj.setShortcut(keys[0])
            else:
                action_obj.setShortcut(QKeySequence())

        # Menu/file/mode action shortcuts
        for action_key, action_obj in [
            ("select_mode", self.actions.selectMode),
            ("create_brush_mode", self.actions.createBrushMode),
            ("create_ai_mask_mode", self.actions.createAiMaskMode),
            ("create_rectangle_mode", self.actions.createBoxAiMaskMode),
            ("create_ai_boundary_mode", self.actions.createAiBoundaryMode),
            ("erase_mode", self.actions.createBoxEraseMode),
            ("create_watershed_3d_mode", self.actions.createWatershed3dMode),
            ("open", self.actions.open),
            ("close", self.actions.close),
            ("quit", self.actions.quit),
            ("save", self.actions.saveMask),
            ("delete_file", self.actions.deleteFile),
            ("save_to", self.actions.changeOutputDir),
            ("open_prev", self.actions.openPrevImg),
            ("open_next", self.actions.openNextImg),
            ("undo", self.actions.undo),
            ("redo", self.actions.redo),
            ("undo_last_point", self.actions.undoLastPoint),
            ("toggle_keep_prev_mode", self.actions.toggleKeepPrevMode),
        ]:
            _apply_action_shortcut(action_obj, action_key)

        # QShortcut objects for label workflow, focus, 3D, escape
        for name, key in [
            ("_sc_verify", sc.get("verify_label", "F")),
            ("_sc_revert", sc.get("revert_label", "R")),
            ("_sc_hide", sc.get("hide_verified", "H")),
            ("_sc_solo", sc.get("solo_label", "S")),
            ("_sc_show_all", sc.get("show_all_labels", "Shift+S")),
            ("_sc_search", sc.get("focus_label_search", "Ctrl+F")),
            ("_sc_brush_label", sc.get("focus_brush_label", "Ctrl+L")),
            ("_sc_set_brush_label_0", sc.get("set_brush_label_0", "0")),
            ("_sc_3d", sc.get("toggle_3d", "Ctrl+3")),
            ("_sc_escape", sc.get("escape", "Escape")),
        ]:
            obj = getattr(self, name, None)
            if obj is not None:
                obj.setKey(QKeySequence(key) if key else QKeySequence())

        reject_keys = sc.get("reject_label", ["Delete", "Backspace"])
        rlist = reject_keys if isinstance(reject_keys, (list, tuple)) else [reject_keys]
        if hasattr(self, "_sc_delete") and self._sc_delete:
            self._sc_delete.setKey(QKeySequence(rlist[0]) if rlist else QKeySequence())
        if hasattr(self, "_sc_backspace") and self._sc_backspace:
            self._sc_backspace.setKey(
                QKeySequence(rlist[1]) if len(rlist) > 1 else QKeySequence()
            )
        commit_keys = sc.get("commit", ["Ctrl+Return", "Meta+Return"])
        clist = commit_keys if isinstance(commit_keys, (list, tuple)) else [commit_keys]
        if hasattr(self, "_sc_commit") and self._sc_commit:
            self._sc_commit.setKey(QKeySequence(clist[0]) if clist else QKeySequence())
        if hasattr(self, "_sc_commit_mac") and self._sc_commit_mac:
            self._sc_commit_mac.setKey(
                QKeySequence(clist[1]) if len(clist) > 1 else QKeySequence()
            )

    def _shortcut_escape(self):
        """Escape: Exit solo mode, clear AI mask points, or switch to select mode."""
        if not self._shortcutAllowed():
            return
        if hasattr(self, "visibilityManager") and self.visibilityManager.is_solo_mode():
            self._onShowAllRequested()
            return
        if (
            self.canvas.createMode == "ai_mask"
            and self.canvas.current is not None
        ):
            self.canvas.current = None
            self.canvas.drawingPolygon.emit(False)
            self.canvas.update()
            return
        self.actions.selectMode.trigger()
    
    def _shortcut_prevSlice(self):
        if not self._shortcutAllowed():
            return
        self.openPrevImg()
    
    def _shortcut_nextSlice(self):
        if not self._shortcutAllowed():
            return
        self.openNextImg()
    
    def _shortcut_setViewAxis(self, axis: int):
        if not self._shortcutAllowed():
            return
        if hasattr(self, 'viewSelection'):
            self.viewSelection.setCurrentIndex(axis)
    
    def _shortcut_verifyLabel(self):
        if not self._shortcutAllowed():
            return
        self.verifySelectedLabel()
    
    def _shortcut_revertLabel(self):
        if not self._shortcutAllowed():
            return
        self.revertSelectedLabel()
    
    def _shortcut_rejectLabel(self):
        if not self._shortcutAllowed():
            return
        self.rejectSelectedLabel()
    
    def _shortcut_commit(self):
        # Commit should work even when typing (it's Ctrl+Enter)
        self.commitChanges()
    
    def _shortcut_toggleHideVerified(self):
        if not self._shortcutAllowed():
            return
        if hasattr(self, 'hideVerifiedCheckbox'):
            current_state = self.hideVerifiedCheckbox.isChecked()
            self.hideVerifiedCheckbox.setChecked(not current_state)
    
    def _shortcut_soloLabel(self):
        if not self._shortcutAllowed():
            return
        self._onSoloCurrentFromButton()
    
    def _shortcut_showAll(self):
        if not self._shortcutAllowed():
            return
        self._onShowAllRequested()
    
    def _shortcut_setBrushLabel0(self):
        """Set brush label ID to 0."""
        if not self._shortcutAllowed():
            return
        if hasattr(self, 'brush_label_input'):
            self.brush_label_input.setText("0")

    def _shortcut_focusLabelSearch(self):
        if hasattr(self, 'labelSearchBox'):
            self.labelSearchBox.setFocus()
            self.labelSearchBox.selectAll()
    
    def _shortcut_focusBrushLabel(self):
        if hasattr(self, 'brush_label_input'):
            self.brush_label_input.setFocus()
            self.brush_label_input.selectAll()
    
    def _shortcut_toggle3D(self):
        if hasattr(self, 'checkBox3DRendering'):
            current_state = self.checkBox3DRendering.isChecked()
            self.checkBox3DRendering.setChecked(not current_state)
    
    def keyPressEvent(self, event):
        """
        Handle keyboard events that need special logic beyond QShortcut.
        Most shortcuts (including Escape) are handled via _installShortcuts().
        """
        super().keyPressEvent(event)


class InterpolateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, start_slice=-1, end_slice=-1, max_slice=100, target_label="10000"):
        super(InterpolateDialog, self).__init__(parent)
        self.setWindowTitle("Fill Between Slices")

        # UI Elements
        self.start_slice_label = QtWidgets.QLabel("Start Slice:")
        self.start_slice_spinbox = QtWidgets.QSpinBox()
        self.start_slice_spinbox.setRange(0, max_slice)
        self.start_slice_spinbox.setValue(start_slice)

        self.end_slice_label = QtWidgets.QLabel("End Slice:")
        self.end_slice_spinbox = QtWidgets.QSpinBox()
        self.end_slice_spinbox.setRange(0, max_slice)
        self.end_slice_spinbox.setValue(end_slice) # Default: next 10 frames

        self.target_label_label = QtWidgets.QLabel("Target Label:")
        self.target_label_input = QtWidgets.QLineEdit()
        self.target_label_input.setPlaceholderText("Enter label ID to interpolate")
        self.target_label_input.setText(target_label)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Layout
        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.start_slice_label, self.start_slice_spinbox)
        layout.addRow(self.end_slice_label, self.end_slice_spinbox)
        layout.addRow(self.target_label_label, self.target_label_input)
        layout.addWidget(self.button_box)

    def getValues(self):
        """Return user input values"""
        return (
            self.start_slice_spinbox.value(),
            self.end_slice_spinbox.value(),
            self.target_label_input.text()
        )
