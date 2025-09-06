# labelme/app.py

# -*- coding: utf-8 -*-
import threading
import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser
import tifffile as tiff
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
from em_util.seg import seg_to_iou

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt
import vtk
from vtk.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from labelme import PY2
from labelme import __appname__
from labelme import ai
from labelme.ai import MODELS
from labelme.config import get_config
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
from labelme.widgets import ZoomWidget
from labelme.utils import compute_tiff_sam_feature, compute_points_from_mask
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

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera



def process_mask(label, mask_data, slice_id):
    """
    Process a single label to create a mask shape.
    """
    if label == 0:
        return None  # Skip the background
    mask = mask_data == label
    y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)

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
        mask=mask[y1 : y2 + 1, x1 : x2 + 1],
    )
    return drawing_shape
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

def numpy_to_vtk_image( data: np.ndarray):
    """
    Convert a 3D numpy array to vtkImageData more efficiently.

    Parameters:
        data (np.ndarray): 3D numpy array.

    Returns:
        vtk.vtkImageData: Converted VTK image data.
    """
    # Ensure the numpy array is contiguous in memory
    data = np.ascontiguousarray(data)

    # Create a vtkImageData object
    vtk_image = vtk.vtkImageData()
    depth, height, width = data.shape
    vtk_image.SetDimensions(width, height, depth)

    # allocate 16-bit unsigned scalars (1 component)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
    # Wrap the numpy array into a VTK array
    vtk_array = numpy_support.numpy_to_vtk(num_array=data.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)

    # Set the VTK array as the scalars for the vtkImageData
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image

def process_label(label, data, smooth_iterations, label_colormap):
    """
    Process a single label: create iso-surface, smooth it, and return actor.
    """
    if label == 0:
        # Skip background (label 0)
        return None

    # Create a binary mask for the current label
    label_data = data.copy()
    label_data[label_data != label] = 0

    # Convert the binary mask to vtkImageData
    vtk_image = numpy_to_vtk_image(label_data)

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


    def update_crosshair_position(self, center_point, data_shape):
        """Update crosshair position and ensure it is visible."""
        if not self.crosshair_actors: # Return if not created yet
            return
        depth, height, width = data_shape
        x, y, z = center_point

        # Update sphere position
        self._crosshair_sources['sphere'].SetCenter(x, y, z)

        # Update the positions of the three lines
        self._crosshair_sources['x'].SetPoint1(0, y, z)
        self._crosshair_sources['x'].SetPoint2(width, y, z)

        self._crosshair_sources['y'].SetPoint1(x, 0, z)
        self._crosshair_sources['y'].SetPoint2(x, height, z)

        self._crosshair_sources['z'].SetPoint1(x, y, 0)
        self._crosshair_sources['z'].SetPoint2(x, y, depth)

        # Ensure all crosshair actors are visible
        if not self.crosshair_actors[0].GetVisibility():
            for actor in self.crosshair_actors:
                actor.SetVisibility(True)

        self.vtkWidget.GetRenderWindow().Render()

    def toggle_label_visibility(self, label, visible):
        """
        Toggle the visibility of a specified label in the 3D rendered scene.

        Parameters:
            label (int): The label value to show or hide.
            visible (bool): True to show the label, False to hide it.
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

        # Refresh the render window to apply changes
        self.vtkWidget.GetRenderWindow().Render()

    def add_grid(self, data: np.ndarray):
        """
        Add a coordinate grid to the 3D scene based on the input data's shape.

        Parameters:
            data (np.ndarray): 3D numpy array to determine grid bounds.
        """
        # Get the bounds from the data shape
        depth, height, width = data.shape
        bounds = [0, width, 0, height, 0, depth]  # x, y, z ranges
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

    def update_surface_with_smoothing(self, data: np.ndarray, smooth_iterations=20):
        """
        Extract and display the 3D surface (iso-surface) of the given data,
        with smoothing applied to the surface. Each label will have a unique color.
        """
        print("Updating 3D surface with smoothing...")

        # Get unique labels in the segmentation data
        unique_labels = np.unique(data)
        print(f"Unique labels: {unique_labels}")

        # Clear previous actors to avoid overlaps
        self.renderer.RemoveAllViewProps()

        # Step 1: Process each label in parallel using ThreadPoolExecutor
        label_colormap = LABEL_COLORMAP  # Define your colormap
        actors = []

        with ThreadPoolExecutor() as executor:
            # Submit tasks for parallel processing
            futures = [
                executor.submit(process_label, label, data, smooth_iterations, label_colormap)
                for label in unique_labels
            ]

            # Collect results as they complete
            for future in futures:
                actor = future.result()
                if actor is not None:
                    actors.append(actor)

        # Step 2: Add actors to the renderer
        for actor in actors:
            self.renderer.AddActor(actor)
        
        for actor in self.crosshair_actors:
                self.renderer.AddActor(actor)
        # Step 3: Add coordinate grid to the renderer
        self.add_grid(data)

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
        # ---------- Create multiple toolbars ----------
        self.file_toolbar = QtWidgets.QToolBar('File && Nav', self)
        self.addToolBar(Qt.TopToolBarArea, self.file_toolbar)
        self.file_toolbar.setObjectName("fileToolbar")
        self.addToolBarBreak()

        self.draw_toolbar = QtWidgets.QToolBar('Draw', self)
        self.addToolBar(Qt.TopToolBarArea, self.draw_toolbar)
        self.draw_toolbar.setObjectName("drawToolbar")
        self.addToolBarBreak()

        self.view_toolbar = QtWidgets.QToolBar('View && Misc', self)
        self.view_toolbar.setObjectName("viewToolbar")
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)

        # Use a unified compact style
        for tb in (self.file_toolbar, self.draw_toolbar, self.view_toolbar):
            tb.setIconSize(QtCore.QSize(18, 18))
            tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            tb.setMovable(False)
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

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
        
        # Add sorting control buttons
        sort_layout = QtWidgets.QHBoxLayout()
        sort_asc_btn = QtWidgets.QPushButton("↑ Size")
        sort_asc_btn.setToolTip("Sort by voxel size (ascending)")
        sort_asc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_voxel_size(ascending=True))
        
        sort_desc_btn = QtWidgets.QPushButton("↓ Size")
        sort_desc_btn.setToolTip("Sort by voxel size (descending)")
        sort_desc_btn.clicked.connect(lambda: self.uniqLabelList.sort_by_voxel_size(ascending=False))
        
        sort_layout.addWidget(sort_asc_btn)
        sort_layout.addWidget(sort_desc_btn)
        sort_layout.addStretch()
        
        label_layout.addLayout(sort_layout)
        label_layout.addWidget(self.uniqLabelList)

        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(label_container)  # Use container widget

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
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

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["label_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        
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
        self.label_input.setPlaceholderText("Label")
        self.delete_label_button = QPushButton("Delete Label", self)
        self.delete_label_button.clicked.connect(self.delete_label)
        self.split_label_button = QPushButton("Split Label", self)
        self.split_label_button.clicked.connect(self.split_label)
        col1_layout.addWidget(self.label_input)
        col1_layout.addWidget(self.delete_label_button)
        col1_layout.addWidget(self.split_label_button)
        top_h_layout.addLayout(col1_layout)

        # Column 2
        col2_layout = QVBoxLayout()
        self.find_connected_slice_input = QLineEdit(self)
        self.find_connected_slice_input.setPlaceholderText("Label ID")

        find_buttons_layout = QHBoxLayout()
        self.find_fm_button = QPushButton("Find FM", self)  # Renamed
        self.find_fm_button.clicked.connect(self.find_connected_slice)
        self.waterz_button = QPushButton("waterz", self)    # New button
        self.waterz_button.clicked.connect(self.apply_watershed) # Connect new functionality
        find_buttons_layout.addWidget(self.find_fm_button)
        find_buttons_layout.addWidget(self.waterz_button)

        # 3D Watershed UI controls
        watershed_3d_layout = QHBoxLayout()
        self.watershed_3d_label_input = QLineEdit(self)
        self.watershed_3d_label_input.setPlaceholderText("Auto-detected from seeds")
        self.watershed_3d_label_input.setReadOnly(True)  # Set read-only
        self.watershed_3d_clear_button = QPushButton("Clear Seeds", self)
        self.watershed_3d_apply_button = QPushButton("Apply 3D Watershed", self)
        
        self.watershed_3d_clear_button.clicked.connect(self.clear_watershed_seeds)
        self.watershed_3d_apply_button.clicked.connect(self.apply_3d_watershed)
        
        watershed_3d_layout.addWidget(QLabel("3D Watershed (Auto Label):"))
        watershed_3d_layout.addWidget(self.watershed_3d_label_input)
        watershed_3d_layout.addWidget(self.watershed_3d_clear_button)
        watershed_3d_layout.addWidget(self.watershed_3d_apply_button)

        # Horizontal layout for Prev/Next buttons
        nav_buttons_layout = QHBoxLayout()
        self.find_prev_button = QPushButton("Prev", self)
        self.find_next_button = QPushButton("Next", self)
        self.find_prev_button.clicked.connect(self.find_prev_connected_slice)
        self.find_next_button.clicked.connect(self.find_next_connected_slice)
        nav_buttons_layout.addWidget(self.find_prev_button)
        nav_buttons_layout.addWidget(self.find_next_button)

        # Add all widgets to the second column layout
        col2_layout.addWidget(self.find_connected_slice_input)
        col2_layout.addLayout(find_buttons_layout) # Add layout containing Find FM and waterz
        col2_layout.addLayout(nav_buttons_layout)
        col2_layout.addLayout(watershed_3d_layout)  # Add 3D watershed controls
        top_h_layout.addLayout(col2_layout)
        main_v_layout.addLayout(top_h_layout)


        label_ops_action = QWidgetAction(self)
        label_ops_action.setDefaultWidget(label_ops_widget)


        # --- Begin vertical Merge Label widget ---
        merge_label_widget = QWidget(self)
        # no fixed width: let it size to contents

        v_layout = QVBoxLayout(merge_label_widget)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(2)
        v_layout.setAlignment(Qt.AlignLeft)

        # Row 1: two inputs + arrow
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)

        self.merge_label_input_1 = QLineEdit(self)
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
        self.merge_label_input_2.setPlaceholderText("L2")
        self.merge_label_input_2.setFixedWidth(30)
        input_layout.addWidget(self.merge_label_input_2)

        v_layout.addLayout(input_layout)

        # Row 2: Merge button, left‐aligned
        self.merge_label_button = QPushButton("Merge Labels", self)
        self.merge_label_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.merge_label_button.clicked.connect(self.merge_labels)
        v_layout.addWidget(self.merge_label_button, alignment=Qt.AlignLeft)

        # Wrap in an action and add to your actual toolbar (self.tools)
        merge_labels_action = QWidgetAction(self)
        merge_labels_action.setDefaultWidget(merge_label_widget)
        # --- End updated widget ---

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

        # Create a QWidgetAction to integrate the brush widget into the UI
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
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevTenImg = action(
            self.tr("&Prev 10"),
            self.openPrevTenImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
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

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("box AI-mask"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing Ai mask by rectangles"),
            enabled=False,
        )
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
            lambda: self.canvas.set_ai_model(
                self._get_or_create_ai_model(self._selectAiModelComboBox.currentText()),
                self.embedding_dir
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("Points AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.set_ai_model(
                self._get_or_create_ai_model(self._selectAiModelComboBox.currentText()),
                self.embedding_dir
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        createAiBoundaryMode = action(
            self.tr("AI-Boundary"),
            lambda: self.toggleDrawMode(False, createMode="ai_boundary"),
            None,
            "objects",
            self.tr("Start drawing ai_boundary by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiBoundaryMode.changed.connect(
            lambda: self.canvas.set_ai_model(
                self._get_or_create_ai_model(self._selectAiModelComboBox.currentText()),
                self.embedding_dir
            )
            if self.canvas.createMode == "ai_boundary"
            else None
        )

        eraseMode = action(
            self.tr("Erase mask"),
            lambda: self.toggleDrawMode(False, createMode="erase"),
            None,
            "objects",
            self.tr("Erase mask by rectangles"),
            enabled=False,
        )
        # Add brush mode action
        createBrushMode = action(
            self.tr("Brush Mode"),
            lambda: self.toggleDrawMode(False, createMode="brush"),
            None,
            "objects",
            self.tr("Start freehand drawing with brush"),
            enabled=False,
        )
        createWatershed3dMode = action(
            self.tr("3D Watershed Seeds"),
            lambda: self.toggleDrawMode(False, createMode="watershed_3d"),
            None,
            "objects",
            self.tr("Click to place seed points for 3D watershed"),
            enabled=False,
        )
        selectMode = action(
            self.tr("View/Select"),
            lambda: self.toggleDrawMode(edit=True),  # Call toggleDrawMode(True) to exit drawing
            "V",  # Shortcut key 'V'
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
        self.mode_action_group.addAction(createAiBoundaryMode)
        self.mode_action_group.addAction(createRectangleMode)
        self.mode_action_group.addAction(eraseMode)
        self.mode_action_group.addAction(createBrushMode)
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
        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        # vvv Add Redo action here vvv
        redo = action(
            self.tr("Redo\n"),
            self.redoShapeEdit,
            shortcuts.get("redo", "Ctrl+Y"), # Assume redo shortcut is Ctrl+Y
            "redo",
            self.tr("Redo last undone edit"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
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

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

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
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            undoLastPoint=undoLastPoint,
            undo=undo,
            redo=redo,
            selectMode=selectMode, 
            createMode=createMode,
            createRectangleMode=createRectangleMode,
            createPointMode=createPointMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            createAiBoundaryMode=createAiBoundaryMode,
            eraseMode=eraseMode,
            createBrushMode=createBrushMode,
            createWatershed3dMode=createWatershed3dMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, close, quit),
            tool=(
                selectMode,
                createAiMaskMode, 
                createRectangleMode,
                createAiBoundaryMode, 
                eraseMode, 
                createBrushMode,
                createWatershed3dMode,
                brush_action, 
                label_ops_action,
                merge_labels_action,
            ),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                None,
                undo,
                redo,
                undoLastPoint,
                None,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createRectangleMode,
                createAiMaskMode,
                createPointMode,
                createMode,
                undo,
                undoLastPoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createPointMode,
                createAiPolygonMode,
                createAiMaskMode,
                brightnessContrast,
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
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
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
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Move here", self.moveShape),
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
        model_names = [model.name for model in MODELS]
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

        # Create the main widget for segment all
        segmentall = QtWidgets.QWidgetAction(self)
        segmentallWidget = QtWidgets.QWidget()
        segmentall.setDefaultWidget(segmentallWidget)

        # Use QVBoxLayout for the overall layout
        mainLayout = QtWidgets.QVBoxLayout(segmentallWidget)

        # Reduce padding and spacing for the main layout
        mainLayout.setContentsMargins(5, 5, 5, 5)  # Set smaller margins (left, top, right, bottom)
        mainLayout.setSpacing(5)  # Set smaller spacing between widgets

        # Add label for the model selector (First row)
        segmentallLabel = QtWidgets.QLabel(self.tr("Segmentation Model"))
        segmentallLabel.setAlignment(QtCore.Qt.AlignCenter)
        mainLayout.addWidget(segmentallLabel)

        # Add model selection dropdown (Second row)
        self._segmentallComboBox = QtWidgets.QComboBox()
        mainLayout.addWidget(self._segmentallComboBox)

        # Populate the dropdown with model options
        model_options = ["cellpose", "nnUnet"]  # Available models
        self._segmentallComboBox.addItems(model_options)

        # Set the default model
        default_model = self._config["segment_all"]["default"]
        if default_model in model_options:
            model_index = model_options.index(default_model)
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
        self.update3DButton = QtWidgets.QPushButton(self.tr("Update 3D"))

        self.interpolateButton = QtWidgets.QPushButton(self.tr("Interpolate"))
        buttonLayout.addWidget(self.segmentAllButton)  # Add Segment All button
        buttonLayout.addWidget(self.trackingButton)    # Add Tracking button
        buttonLayout.addWidget(self.interpolateButton)

        #buttonLayout.addWidget(self.update3DButton)

        mainLayout.addLayout(buttonLayout)  # Add button layout to the main layout

        # Connect buttons to their respective actions
        self.segmentAllButton.clicked.connect(self.segmentAll)
        self.trackingButton.clicked.connect(self.tracking)
        self.update3DButton.clicked.connect(self.update3D)
        self.interpolateButton.clicked.connect(self.show_interpolate_dialog)


        # Ai prompt
        self._ai_prompt_widget: QtWidgets.QWidget = AiPromptWidget(
            on_submit=self._submit_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        # ---------- File / Navigation ----------
        utils.addActions(self.file_toolbar,
            (openPrevImg, openNextImg,saveMask))

        # ---------- Draw / Labels ----------
        self.draw_toolbar.addActions([
            createAiMaskMode, 
            createAiBoundaryMode,
            createRectangleMode, 
            eraseMode, 
            createBrushMode,
            createWatershed3dMode,
            selectMode,
        ])
        
        self.draw_toolbar.addAction(brush_action)
        self.draw_toolbar.addAction(label_ops_action)
        self.draw_toolbar.addAction(merge_labels_action)

        # ---------- View / Misc ----------
        self.view_toolbar.addAction(selectAiModel)
        self.view_toolbar.addAction(segmentall)
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

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Initialize cache and threading
        self.sliceCache = {}  # Dictionary to store cached slices
        self.cacheThread = None  # Thread for background caching
        self.cacheRange = 5  # Number of slices to cache before and after the current slice
        self.currentSliceIndex = 0  # Current slice index
        self.currentSliceIndex = 0  # Default slice index
        self.currentViewAxis = 0  # Default axis: 0 = Axial, 1 = Coronal, 2 = Sagittal

        self.viewSelection = QtWidgets.QComboBox()
        self.viewSelection.addItems(["Axial", "Coronal", "Sagittal"])  # 0, 1, 2 respectively
        self.viewSelection.currentIndexChanged.connect(self.updateViewAxis)

        # Create a layout for the selection
        viewControlWidget = QtWidgets.QWidget()
        viewControlLayout = QtWidgets.QHBoxLayout()
        viewControlLayout.addWidget(QtWidgets.QLabel("View:"))
        viewControlLayout.addWidget(self.viewSelection)
        viewControlWidget.setLayout(viewControlLayout)

        # Add widget to toolbar
        viewSelectionAction = QtWidgets.QWidgetAction(self)
        viewSelectionAction.setDefaultWidget(viewControlWidget)
        


        # initialize lastClickedPoint so it always exists
        self.lastClickedPoint = None


        # --- Add a 3D rendering mode toggle ---
        # 1) Track whether to show all labels in 3D
        self.showAll3D = False 

        self.crosshair_center_xy = None

        # 2) Create the checkbox widget
        self.checkBox3DRendering = QtWidgets.QCheckBox("Show All 3D")
        self.checkBox3DRendering.setChecked(self.showAll3D)
        self.checkBox3DRendering.stateChanged.connect(self.on3DRenderingCheckBoxChanged)
        self.checkBox3DRendering.setLayoutDirection(QtCore.Qt.RightToLeft)

        # --- Create a new composite control for View Selection and 3D-related buttons (vertical three-row layout) ---
        
        # 1. Create the outer container widget and its main vertical layout
        view_3d_controls_widget = QtWidgets.QWidget()
        main_v_layout = QtWidgets.QVBoxLayout(view_3d_controls_widget)
        main_v_layout.setContentsMargins(5, 5, 5, 5)
        main_v_layout.setSpacing(2)
        main_v_layout.setAlignment(QtCore.Qt.AlignTop) # Align all controls to top
        
        # 2. Create and add the first row: View Selection
        #    (Ensure self.viewSelection is already created)
        top_row_layout = QtWidgets.QHBoxLayout()
        top_row_layout.addWidget(QtWidgets.QLabel("View:"))
        top_row_layout.addWidget(self.viewSelection)
        main_v_layout.addLayout(top_row_layout)

        # 3. Add the CheckBox as the second row to the main vertical layout
        #    (Ensure self.checkBox3DRendering is already created)
        main_v_layout.addWidget(self.checkBox3DRendering)

        # 4. Add the Update 3D Button as the third row to the main vertical layout
        #    (Ensure self.update3DButton is already created)
        main_v_layout.addWidget(self.update3DButton)
        
        # 5. Wrap this new composite control in a QWidgetAction
        view_3d_controls_action = QtWidgets.QWidgetAction(self)
        view_3d_controls_action.setDefaultWidget(view_3d_controls_widget)

        # 6. Finally, add this new action to the view_toolbar
        self.view_toolbar.addAction(view_3d_controls_action)

        # --- End of new composite control creation ---
        self.label_visibility_states = {}
        self.compute_thread = None
        self.compute_thread_stop_event = None 
        self.embedding_task_queue = None  
        self.ai_model_cache = {}  # Cache for AI model 
        self.recent_label = "10000"  # Store the most recent label for AI operations
        self._sliceLoadTimer = QtCore.QTimer(self)
        self._sliceLoadTimer.setSingleShot(True)
        self._sliceLoadTimer.timeout.connect(self.loadAnnotationsAndMasks)
        self._sliceLoadDelayMs = 120  # try 120–200ms
        self._handling_visibility = False



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
        # 1) First clear existing actions on draw_toolbar
        for act in list(self.draw_toolbar.actions()):
            self.draw_toolbar.removeAction(act)

        # 2) Re-add drawing/label-related tool buttons to draw_toolbar
        utils.addActions(self.draw_toolbar, self.actions.tool)

        # 3) Update the Canvas context menu
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.actions.menu)

        # 4) Update the main window's Edit menu
        self.menus.edit.clear()
        edit_actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createPointMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.createAiBoundaryMode,
        )
        utils.addActions(self.menus.edit, edit_actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isUndoable)
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
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        self.actions.createAiBoundaryMode.setEnabled(True)
        self.actions.eraseMode.setEnabled(True)
        self.actions.createBrushMode.setEnabled(True)
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
        self._update_undo_actions()

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
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.camera_initialized = False
        self.segmentAllModel = None
        self.label_list = [i for i in range(1, MAX_LABEL)]
        self.sliceCache = {}

    def undoShapeEdit(self):
        if not self.canvas.isUndoable:
            return
        self.canvas.undo()
        # Use the fixed loadShapes to efficiently refresh the UI
        self.loadShapes(self.canvas.shapes, replace=True)
        self._update_undo_actions()
        self.setDirty()

    def redoShapeEdit(self):
        if not self.canvas.isRedoable:
            return
        self.canvas.redo()
        self.loadShapes(self.canvas.shapes, replace=True)
        self._update_undo_actions()
        self.setDirty()

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """


    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "erase": self.actions.eraseMode,
            "brush": self.actions.createBrushMode,
            "point": self.actions.createPointMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
            "ai_boundary":self.actions.createAiBoundaryMode,
            "watershed_3d": self.actions.createWatershed3dMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
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

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)


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
        
        if self.canvas.createMode == "erase":
            self.tiffMask[index_tuple] = 0
        elif self.canvas.createMode == "brush":
            self.tiffMask[index_tuple][mask > 0] = int(self.brush_label_input.text())
        else:
            self.tiffMask[index_tuple][mask > 0] = int(label)
        self.actions.saveMask.setEnabled(True)
        self.last_ai_mask_slice = shape.slice_id
        self.updateUniqueLabelListFromEntireMask()


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
            # 3) Update the icon
            self.uniqLabelList.setItemLabel(item, label, rgb)
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
        self.actions.saveMask.setEnabled(True)


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
        if not shapes:  # If there are no shapes, return directly
            if replace:
                self.canvas.loadShapes([], replace=True)
            return
            
        self._noSelectionSlot = True

        # Call minimal operation for each shape during scrolling
        for shape in shapes:
            self.addLabelMinimal(shape)

        # Clear selection
        self._noSelectionSlot = False

        # Load shapes into the canvas - this is user-visible; do it immediately
        self.canvas.loadShapes(shapes, replace=replace)
        
        # Apply critical visibility settings immediately, not via timer
        for shape in shapes:
            is_visible = self.label_visibility_states.get(shape.label, True)
            if not is_visible:
                self.canvas.setShapeVisible(shape, False, update=False)
        
        # Update the canvas once at the end
        self.canvas.update()
        
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

    def _update_undo_actions(self):
        self.actions.undo.setEnabled(self.canvas.isUndoable)
        self.actions.redo.setEnabled(self.canvas.isRedoable)
    
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
                        self.get_current_slice(self.tiffData, pred_slice_index),
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
                    self.actions.saveMask.setEnabled(True)
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
            if prompt_points:
                # Add prompt points to currentAIPromptPoints
                # If createMode is "rectangle", add all prompt points, otherwise add the first prompt point
                if self.canvas.createMode == "rectangle":
                    self.currentAIPromptPoints.append((prompt_points, shape.label))
                else:
                    self.currentAIPromptPoints.append((prompt_points[0], shape.label))
            shape.group_id = group_id
            shape.description = description
            shape.slice_id = self.currentSliceIndex
            print(f"createMode: {self.canvas.createMode}")
            self.addLabel(shape)
            if shape.shape_type == "mask":
                self._update_mask_to_tiffMask(shape)
                # Refresh current slice
                self.openNextImg(nextN=0)
            
            if shape.shape_type == "points": # use these points as the prompt points
                pass
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
            self._update_undo_actions()
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
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
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
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

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
        img = img.astype(np.float32)
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype(np.uint8)
        return img
    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

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
            return False

        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))

        # Check if the file is a TIFF file
        if filename.lower().endswith(('.tiff', '.tif')):
            try:
                # Load the 3D TIFF file
                self.tiffData = tiff.imread(filename)
                for i in range(len(self.tiffData)):
                    self.tiffData[i] = self.normalizeImg(self.tiffData[i])
                print(f"TIFF data shape: {self.tiffData.shape}")
                file_dir = osp.dirname(filename)
                cell_name = osp.basename(filename).split(".")[0]
                model_name = self._selectAiModelComboBox.currentText()
                self.embedding_dir=f"{file_dir}/{cell_name}_embeddings_{model_name}_axis{self.currentViewAxis}"
                model_instance = self._get_or_create_ai_model(model_name)
                if model_instance:
                    self.canvas.set_ai_model(model_instance, self.embedding_dir)

                print(f"Initialize ai model with Embedding dir: {self.embedding_dir}")
                self.currentSliceIndex = 0
                if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < self.tiffData.shape[self.currentViewAxis]:
                    self.status("Starting background embedding calculation...")

                    # --- Create task queue and stop event ---
                    self.embedding_task_queue = queue.Queue()
                    self.compute_thread_stop_event = threading.Event()

                    # --- Fill initial task list (0 -> N) ---
                    num_slices = self.tiffData.shape[self.currentViewAxis]
                    for i in range(num_slices):
                        self.embedding_task_queue.put(i)

                    # --- Start background worker thread ---
                    model_name = self._selectAiModelComboBox.currentText()
                    self.compute_thread = threading.Thread(
                        target=compute_tiff_sam_feature,
                        args=(self.tiffData, model_name, self.embedding_dir, self.currentViewAxis, self.embedding_task_queue, self.compute_thread_stop_event),
                        daemon=True
                    )
                    self.compute_thread.start()
                if self.tiffData.ndim == 3:
                    # Assuming the 3D image is a stack of 2D images, take the first slice
                    self.imageData = self.normalizeImg(self.get_current_slice(self.tiffData,0))  # Load the first slice for display
                    self.imagePath = filename
                    h, w = self.imageData.shape
                    bytes_per_line = self.imageData.strides[0]  # For uint8 arrays, usually equals w
                    self.image = QImage(
                        self.imageData.data,    # Pixel buffer
                        w,                      # width
                        h,                      # height
                        bytes_per_line,         # bytesPerLine
                        QImage.Format_Grayscale8,
                    )
                    self.actions.openNextImg.setEnabled(True)
                    self.actions.openPrevImg.setEnabled(True)
                else:
                    self.errorMessage(
                        self.tr("Error opening file"),
                        self.tr("Only 3D TIFF files with grayscale slices are supported."),
                    )
                    return False
            except Exception as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr("Failed to read TIFF file: %s") % str(e),
                )
                return False
        else:
            # Fallback for other image formats
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
            self.image = QImage.fromData(self.imageData)

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
            return False

        # Load the image onto the canvas
        self.canvas.loadPixmap(QPixmap.fromImage(self.image),slice_id=self.currentSliceIndex)
        self.filename = filename
        
        # Load JSON annotation if exists
        self.annotation_json = filename.replace(".tiff", ".json").replace(".tif", ".json")
        if os.path.exists(self.annotation_json):
            try:
                with open(self.annotation_json, "r") as f:
                    self.tiffJsonAnno = json.load(f)

                shapes = []
                # Parse the new JSON format
                slice_key = str(0)  # Assuming first slice for now
                if slice_key in self.tiffJsonAnno and 'rectangle' in self.tiffJsonAnno[slice_key]:
                    for rect in self.tiffJsonAnno[slice_key]['rectangle']:
                        x1, y1, x2, y2, label = rect
                        shape = Shape(
                            label=label,
                            shape_type="rectangle",
                            description="",
                            slice_id=self.currentSliceIndex
                        )
                        # Add rectangle points
                        shape.addPoint(QtCore.QPointF(x1, y1))
                        shape.addPoint(QtCore.QPointF(x2, y2))
                        shapes.append(shape)

                self.canvas.storeShapes()
                self.loadShapes(shapes, replace=False)
                self.status(f"Loaded annotations from {self.annotation_json}")
            except Exception as e:
                self.errorMessage(
                    self.tr("Error loading annotations"),
                    self.tr("Failed to read JSON file: %s") % str(e),
                )
       
        # Load the mask file if it exists
        self.tiff_mask_file = filename.replace(".tif", "_mask.tif")
        if os.path.exists(self.tiff_mask_file) and self.tiff_mask_file != filename:
            try:
                self.tiffMask = tiff.imread(self.tiff_mask_file).astype(np.uint16)
                self.updateUniqueLabelListFromEntireMask()
                mask_data = self.get_current_slice(self.tiffMask, 0)
                print(f"Mask data shape: {mask_data.shape}")

                shapes = []
                # Loop through each label in the mask
                for label in np.unique(mask_data):
                    if label == 0:
                        continue  # Skip the background
                    mask = mask_data == label
                    y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)

                    drawing_shape = Shape(
                        label=str(label),
                        shape_type="mask",
                        description=f"Mask for label {label}",
                        slice_id=self.currentSliceIndex
                    )
                    drawing_shape.setShapeRefined(
                        shape_type="mask",
                        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                        point_labels=[1, 1],
                        mask=mask[y1 : y2 + 1, x1 : x2 + 1],
                    )
                    shapes.append(drawing_shape)

                self.canvas.storeShapes()
                self.loadShapes(shapes, replace=False)
                self.status(f"Loaded mask annotations from {self.tiff_mask_file}")
            except Exception as e:
                self.errorMessage(
                    self.tr("Error loading mask file"),
                    self.tr("Failed to read mask file: %s") % str(e),
                )
       
        self.setClean()
        self.canvas.setEnabled(True)
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
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
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

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
        for label in np.unique(mask_data):
            if label == 0:
                continue  # Skip background
            # Build shapes only for labels that are globally visible
            if not self.label_visibility_states.get(str(label), True):
                continue

            y1, y2, x1, x2, roi_mask = self._fast_bbox_and_roi(mask_data, int(label))
            drawing_shape = Shape(
                label=str(label),
                shape_type="mask",
                description=f"Mask for label {label}",
                slice_id=slice_index,
            )
            drawing_shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=roi_mask,
            )
            shapes.append(drawing_shape)

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
        """
        self.currentViewAxis = index
        self.currentSliceIndex = 0  # Reset to the first slice in new view
        self.loadFile(self.filename)
        self.updateDisplayedSlice()

    def updateDisplayedSlice(self):
        """
        Update the displayed slice based on the selected view plane.
        """
        if self.tiffData is None:
            return

        slice_data = self.get_current_slice(self.tiffData)

        # Normalize and display the selected slice
        slice_data = self.normalizeImg(slice_data)
        h, w = slice_data.shape
        bytes_per_line = slice_data.strides[0]
        image = QtGui.QImage(
            slice_data.data, w, h,
            bytes_per_line, QtGui.QImage.Format_Grayscale8
        )
        pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.canvas.loadPixmap(pixmap, slice_id=self.currentSliceIndex)
        
        if hasattr(self, 'tiffData') and self.tiffData is not None:
            # If user never clicked, default crosshair to slice center
            if self.crosshair_center_xy is None:
                h, w = self.get_current_slice(self.tiffData).shape[:2]
                center_x, center_y = w / 2, h / 2
            else:
                center_x, center_y = self.crosshair_center_xy

            # --- Use the helper to get correct 3D coordinates ---
            canvas_center_pos = QtCore.QPointF(center_x, center_y)
            point_3d = self._get_3d_point_from_2d(canvas_center_pos)
            # ----------------------------------------

            self.vtk_widget.update_crosshair_position(point_3d, (self.tiffData.shape[2], self.tiffData.shape[1], self.tiffData.shape[0]))

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
            # Check if the previous slice exists
            if self.currentSliceIndex - nextN >= 0:
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

    def openNextImg(self, _value=False, load=True, nextN=1):
        """
        Navigate to the next slice, using cached data if available.
        Automatically trigger caching for surrounding slices.
        """
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if hasattr(self, "tiffData") and self.tiffData is not None:
            # Check if the next slice exists
            max_slices = self.tiffData.shape[self.currentViewAxis]
            if self.currentSliceIndex + nextN < max_slices:
                self.currentSliceIndex += nextN  # Update to the next slice index
                self.updateDisplayedSlice()

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

    def updateUniqueLabelListFromEntireMask(self):
        """
        Sync the unique label list based on the entire tiffMask.
        This method adds missing labels and removes labels no longer present in the mask,
        ensuring the list always reflects the full set of labels in the 3D volume.
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.uniqLabelList.clear() # Clear the list if there is no mask
            return

        # First update voxel count info
        self.uniqLabelList.set_tiff_mask(self.tiffMask)

        # 1. Get all non-zero unique labels from the 3D mask
        #    Use a set to improve subsequent operations
        labels_in_mask = {str(l) for l in np.unique(self.tiffMask) if l != 0}

        # 2. Get all labels currently in the UI list
        labels_in_widget = set()
        for i in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(i)
            labels_in_widget.add(item.data(QtCore.Qt.UserRole))

        # 3. Add new labels: labels present in mask but missing in UI
        labels_to_add = labels_in_mask - labels_in_widget
        if labels_to_add:
            # Use natsort.natsorted to add labels in natural order (1, 2, 10 instead of 1, 10, 2)
            import natsort
            for label in natsort.natsorted(list(labels_to_add)):
                # This helper creates and adds the item automatically
                rgb = self._get_rgb_by_label(label)
                item = self.uniqLabelList.createItemFromLabel(label, rgb=rgb, checked=True)
                self.uniqLabelList.addItem(item)

        # 4. Remove old labels: present in UI but disappeared from mask
        labels_to_remove = labels_in_widget - labels_in_mask
        if labels_to_remove:
            for label in labels_to_remove:
                item = self.uniqLabelList.findItemByLabel(label)
                if item:
                    # takeItem removes the specified item from the list
                    self.uniqLabelList.takeItem(self.uniqLabelList.row(item))


    def loadAnnotationsAndMasks(self):
        """
        Load annotations and masks for the current slice with optimizations.
        """
        shapes = []

        # Load mask data for the current slice
        if self.tiffMask is not None:
            mask_data = self.get_current_slice(self.tiffMask)
            unique_labels = np.unique(mask_data)

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_mask, label, mask_data, self.currentSliceIndex)
                    for label in unique_labels
                ]
                for future in futures:
                    result = future.result()
                    if result is not None:
                        shapes.append(result)

        # # Before loading shapes into the canvas, set each shape's visibility based on the global state
        # for shape in shapes:
        #     # Get visibility from the global state dict; default to True (visible) if not recorded
        #     is_visible = self.label_visibility_states.get(shape.label, True)
        #     shape.visible = is_visible  # Set the shape object's property directly

        # Update the canvas with the loaded annotations and masks
        #self.canvas.storeShapes()
        #self.loadShapes(shapes, replace=False)
        #if self.canvas.createMode != "erase":
        self.loadShapesFromTiff(shapes, replace=True)
        self.setClean()
        self.canvas.setEnabled(True)
        self.status(f"Loaded slice {self.currentSliceIndex}/{self.tiffData.shape[0]}")
    
    def wheelEvent(self, event):
        """
        Mouse wheel event handler. Used to scroll through TIFF slices.
        """
        # Get the global cursor position
        cursor_pos = QtGui.QCursor.pos()
        # Convert to local positions relative to each widget
        scroll_area_pos = self.scrollArea.mapFromGlobal(cursor_pos)
        if hasattr(self, "tiffData") and self.tiffData is not None and self.scrollArea.rect().contains(scroll_area_pos):
            # Determine wheel direction: up loads previous slice, down loads next slice
            if event.angleDelta().y() > 0:  # Wheel up
                self.openPrevImg()
            else:  # Wheel down
                self.openNextImg()
            event.accept()
        else:
            # If not TIFF data, do other actions or ignore
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

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveMask(self, _value=False):
        """
        Update the mask in a TIFF file using information from a updated JSON file.
        """
        print("save tiff mask")
        tiff.imwrite(self.tiff_mask_file, self.tiffMask,  compression="zlib")
        self.actions.saveMask.setEnabled(False)
        self.currentAIPromptPoints = []
        print(f"Updated TIFF file saved to {self.tiff_mask_file}")

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


    def _fuse_segmentations(self, x_seg, y_seg, filter_size=20, overlap_thresh=0.5):
        y_seg += OFFSET_LABEL
        y_seg[y_seg==OFFSET_LABEL] = 0
        outy_x = seg_to_iou(y_seg, x_seg)
        for pair in outy_x:
            if pair[1] ==0 and pair[2] < filter_size:
                y_seg[y_seg==pair[0]] = 0
            if pair[1] !=0 and pair[4] / (pair[2] + pair[3] - pair[4]) > overlap_thresh:
                y_seg[y_seg==pair[0]] = pair[1]
        x_seg[(y_seg != 0) & (x_seg==0)] = y_seg[(y_seg != 0) & (x_seg==0)]

        # Reset the label > OFFSET_LABEL
        for label in np.unique(x_seg):
            if label > OFFSET_LABEL:
                x_seg[x_seg==label] = self.label_list.pop(0)
        print(f"unqiue labels {np.unique(x_seg)}")
        return x_seg
    def segmentAll(self):
        print(f"Segmenting all in current slice {self.currentSliceIndex} using model {self._segmentallComboBox.currentText()}")
        if not hasattr(self, 'tiffData') or self.tiffData is None or not hasattr(self, 'imageData') or self.imageData is None:
            print("No image data available.")
            return
        model_name = self._segmentallComboBox.currentText()
        if not hasattr(self, 'segmentAllModel') or self.segmentAllModel is None or self.segmentAllModel.name != model_name:
            model = [model for model in MODELS if model.name == model_name][0]
            self.segmentAllModel = model()
        pred_mask = self.segmentAllModel.predict(self.imageData)
        # Get the index tuple for the current slice using dynamic slicing.
        idx = self.get_current_slice_index(self.tiffMask)
        if self.tiffMask is None and self.tiffData is not None:
            self.tiffMask = np.zeros(self.tiffData.shape, dtype=np.uint16)
        if np.sum(self.get_current_slice(self.tiffMask)) != 0:
            self.label_list = list(set(self.label_list) - set(np.unique(self.tiffMask)))
            # fuse seg with existing mask
            self.tiffMask[idx] = self._fuse_segmentations(self.tiffMask[idx], pred_mask)
        else:
            self.tiffMask[idx] = pred_mask

        # Set save mask button enabled
        self.actions.saveMask.setEnabled(True)
        self.updateUniqueLabelListFromEntireMask()
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
        """
        self.lastClickedPoint = point

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return
            
        self.crosshair_center_xy = (point.x(), point.y())

        # --- Use helper to compute correct 3D coordinates ---
        point_3d = self._get_3d_point_from_2d(point)
        # ----------------------------------------

        # Update crosshair in 3D view
        # Note: self.tiffData.shape order is (D, H, W) -> (Z, Y, X)
        # while vtk_widget expects (X, Y, Z)
        self.vtk_widget.update_crosshair_position(point_3d, (self.tiffData.shape[2], self.tiffData.shape[1], self.tiffData.shape[0]))

        # If single-label rendering mode is active, refresh the 3D view
        if not self.showAll3D:
            self.update3D()
            
        # Move the 3D camera focus to the new point
        self.vtk_widget.center_camera_on_point(point_3d)

    def on3DRenderingCheckBoxChanged(self, state: int):
        """
        Handle checkbox state changes:
        - True: render all labels in 3D
        - False: render only the label at the last clicked canvas point
        """
        self.showAll3D = (state == QtCore.Qt.Checked)
        # Immediately refresh the 3D view
        self.update3D()
    
    def update3D(self):
        """
        Update the 3D view based on showAll3D flag:
        - If True: render the full mask volume
        - If False: render only the mask for the last clicked label
        """
        self.status("Updating 3D view of segmentation")
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            print("No mask data available.")
            return

        if self.showAll3D:
            volume = self.tiffMask
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
            # Build a volume that contains only this label
            volume = np.where(self.tiffMask == label, label, 0).astype(self.tiffMask.dtype)

        # Call the existing VTK update routine
        self.vtk_widget.update_surface_with_smoothing(
            volume, smooth_iterations=50
        )
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

    def merge_labels(self):
        try:
            label1 = int(self.merge_label_input_1.text())
            label2 = int(self.merge_label_input_2.text())
            if not hasattr(self, 'tiffMask') or self.tiffMask is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available.")
                return
            self.tiffMask[self.tiffMask == label1] = label2
            self.actions.saveMask.setEnabled(True)
            self.updateUniqueLabelListFromEntireMask()

            # Refresh current slice
            self.openNextImg(nextN=0)
            QtWidgets.QMessageBox.information(self, "Success", f"Label {label1} merged into {label2}.")
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
            self.actions.saveMask.setEnabled(True)
            self.updateUniqueLabelListFromEntireMask()

            # Refresh current slice
            self.openNextImg(nextN=0)
            QtWidgets.QMessageBox.information(self, "Success", f"Label {label_to_delete} deleted.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer label.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

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

        # 3) extract only the voxels matching target_label
        mask = self.tiffMask
        roi = (mask == target_label)

        # 4) label all connected components on the binary ROI
        #    returns 0..N where 0 is background, 1..N are components
        cc_map = cc3d.connected_components(roi, connectivity=26)
        
        # [New] 4.5) Filter out connected components smaller than the threshold
        if cc_map.max() > 0: # Only filter when at least one component is found
            # Use cc3d.statistics to efficiently compute voxel counts per component
            stats = cc3d.statistics(cc_map)
            voxel_counts = stats['voxel_counts']
            
            # Find labels of components smaller than threshold (voxel_counts[0] is background)
            small_labels = [label for label, count in enumerate(voxel_counts[1:], 1) if count < size_threshold]

            if small_labels:
                # Use np.isin to efficiently set small components to 0 (background)
                cc_map[np.isin(cc_map, small_labels)] = 0
        
        # [Change] Relabel to ensure filtered labels are contiguous (1, 2, 3, ...)
        # This avoids gaps when assigning new labels later
        final_cc_map, num_components_after_filter = cc3d.connected_components(cc_map, connectivity=26, return_N=True)

        if num_components_after_filter == 0:
            QtWidgets.QMessageBox.information(
                self,
                "No Components",
                f"No connected components larger than {size_threshold} voxels found for label {target_label}."
            )
            # Ensure original ROI region is cleared even if no components are found
            mask[roi] = 0
            self.tiffMask = mask
            self.openNextImg(nextN=0) # Refresh view
            return

        # 5) offset new component IDs so they don't collide with existing labels
        offset = int(mask.max())
        new_mask = mask.copy()
        
        # [Change] Use filtered and relabeled final_cc_map to update new_mask
        # First zero out original ROI to avoid lingering old labels
        new_mask[roi] = 0
        # Then assign new labels only where components exist
        new_mask[final_cc_map > 0] = offset + final_cc_map[final_cc_map > 0]

        # 6) update the in‐memory mask and enable saving
        self.tiffMask = new_mask.astype(mask.dtype)
        self.actions.saveMask.setEnabled(True)
        self.updateUniqueLabelListFromEntireMask()

        # 7) refresh the displayed slice immediately
        self.openNextImg(nextN=0)

        # 8) [Change] Inform the user how many components were created after filtering
        QtWidgets.QMessageBox.information(
            self,
            "Split Completed",
            f"Label {target_label} was split into {num_components_after_filter} components (size >= {size_threshold})."
        )

# labelme/app.py -> class MainWindow

    def apply_watershed(self):
        """
        Applies 2D watershed, then converts each resulting region into its own
        labeled boundary with a controllable thickness.
        """
        try:
            from skimage.feature import peak_local_max
        except ImportError:
            self.errorMessage("Scikit-image Missing", "Please install scikit-image to use this feature (`pip install scikit-image`).")
            return

        try:
            label_to_process = int(self.find_connected_slice_input.text())
        except ValueError:
            self.statusBar().showMessage("Please enter a valid integer label for watershed.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("Mask data not available for watershed.")
            return

        # --- Set boundary thickness directly here ---
        thickness = 2  # You can change this number to adjust boundary thickness (e.g., 1, 2, 3...)
        self.statusBar().showMessage(f"Applying watershed to generate boundaries (thickness: {thickness})...")

        mask_slice = self.get_current_slice(self.tiffMask).copy()
        region_to_split = (mask_slice == label_to_process)
        
        if not np.any(region_to_split):
            self.statusBar().showMessage(f"Label {label_to_process} not found on this slice.")
            return

        # 1. Run watershed to segment the region
        distance = ndi.distance_transform_edt(region_to_split)
        local_maxi = peak_local_max(distance, labels=region_to_split, min_distance=7, exclude_border=False)
        
        if local_maxi.shape[0] < 1:
            self.statusBar().showMessage(f"No distinct centers found for label {label_to_process}.")
            return
            
        markers_mask = np.zeros(distance.shape, dtype=bool)
        markers_mask[tuple(local_maxi.T)] = True
        markers, _ = ndi.label(markers_mask)
        
        ws_labels = watershed(-distance, markers, mask=region_to_split)

        # 2. For each segmented region, compute its boundary and assign new labels
        if ws_labels.max() > 0:
            
            # Get the current maximum label to ensure new labels are unique
            max_existing_label = self.tiffMask.max()
            
            # Iterate over all new regions from watershed (ws_labels values 1, 2, 3...)
            for i in range(1, ws_labels.max() + 1):
                # a. Extract a single region
                single_region_mask = (ws_labels == i)
                
                # b. Compute 1-pixel boundary of the region
                eroded_mask = ndi.binary_erosion(single_region_mask)
                boundary = single_region_mask & ~eroded_mask
                
                # c. Thicken boundary if needed
                if thickness > 1 and np.any(boundary):
                    boundary = ndi.binary_dilation(boundary, iterations=thickness - 1)
                
                # d. Set boundary to 0
                mask_slice[boundary] = 0

            # 3. Update the modified slice back into the 3D mask
            idx = self.get_current_slice_index(self.tiffMask)
            self.tiffMask[idx] = mask_slice

            # 4. Refresh UI
            self.actions.saveMask.setEnabled(True)
            self.updateUniqueLabelListFromEntireMask()
            self.loadAnnotationsAndMasks()
            self.openNextImg(nextN=0)  # Refresh current slice display
            self.statusBar().showMessage(f"Generated boundaries for {ws_labels.max()} new instances.")
        else:
            self.statusBar().showMessage("Watershed did not produce any regions.")

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
        
        # Get the label value at the clicked position
        clicked_label = self.canvas.getLabelAtPosition(x, y, slice_idx, self.tiffMask)
        
        if clicked_label is None or clicked_label == 0:
            self.statusBar().showMessage("Please click on a labeled region (not background).")
            return
        
        # Check if this is the first seed point
        if not self.canvas.watershed_seed_points:
            # First seed point: set target label
            self.canvas.watershed_auto_label = clicked_label
            self.watershed_3d_label_input.setText(str(clicked_label))
            
            # Add seed point
            seed_point = {
                'x': x,
                'y': y,
                'slice_idx': slice_idx,
                'label': clicked_label
            }
            self.canvas.watershed_seed_points.append(seed_point)
            self.canvas.update()
            
            self.statusBar().showMessage(f"Added first seed point for label {clicked_label}.")
            
        else:
            # Check whether the new seed point is on the same label
            if clicked_label != self.canvas.watershed_auto_label:
                self.statusBar().showMessage(
                    f"Error: Clicked on label {clicked_label}, but previous seeds are on label {self.canvas.watershed_auto_label}. "
                    f"Please click 'Clear Seeds' and start over, or click on label {self.canvas.watershed_auto_label}."
                )
                return
            
            # Add seed point
            seed_point = {
                'x': x,
                'y': y,
                'slice_idx': slice_idx,
                'label': clicked_label
            }
            self.canvas.watershed_seed_points.append(seed_point)
            self.canvas.update()
            
            self.statusBar().showMessage(f"Added seed point #{len(self.canvas.watershed_seed_points)} for label {clicked_label}.")
        
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
            # Get the 3D region for the target label
            target_region = (self.tiffMask == target_label)
            
            if not np.any(target_region):
                self.statusBar().showMessage(f"Label {target_label} not found in the mask.")
                return

            # 🚀 Key optimization: compute bounding box and extract subregion
            # Compute 3D bounding box
            bbox = self.compute_bbox_3d(target_region)
            if bbox is None:
                self.statusBar().showMessage("Failed to compute bounding box.")
                return
                
            z_min, z_max, y_min, y_max, x_min, x_max = bbox
            
            # Add padding to ensure boundary integrity
            padding = 5  # Adjustable if needed
            z_min = max(0, z_min - padding)
            z_max = min(self.tiffMask.shape[0], z_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(self.tiffMask.shape[1], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(self.tiffMask.shape[2], x_max + padding)
            
            # Display bounding box info
            subvolume_size = f"{z_max-z_min+1}x{y_max-y_min+1}x{x_max-x_min+1}"
            original_size = f"{self.tiffMask.shape[0]}x{self.tiffMask.shape[1]}x{self.tiffMask.shape[2]}"
            self.statusBar().showMessage(f"Processing subvolume {subvolume_size} from original {original_size}...")
            
            # Extract subregion
            target_subregion = target_region[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Create seed point markers within the subregion
            markers_sub = np.zeros_like(target_subregion, dtype=np.int32)
            for i, seed in enumerate(seed_points):
                z, y, x = seed['slice_idx'], seed['y'], seed['x']
                # Convert to subregion coordinates
                z_sub = z - z_min
                y_sub = y - y_min
                x_sub = x - x_min
                if (0 <= z_sub < target_subregion.shape[0] and 
                    0 <= y_sub < target_subregion.shape[1] and 
                    0 <= x_sub < target_subregion.shape[2]):
                    markers_sub[z_sub, y_sub, x_sub] = i + 1
            
            # 🚀 Run watershed on the subregion (significantly reduced computation)
            distance_sub = ndi.distance_transform_edt(target_subregion)
            from skimage.segmentation import watershed
            ws_labels_sub = watershed(-distance_sub, markers_sub, mask=target_subregion)
            
            # Map the result back to original coordinates
            ws_labels = np.zeros_like(self.tiffMask, dtype=np.int32)
            ws_labels[z_min:z_max, y_min:y_max, x_min:x_max] = ws_labels_sub

            # Update mask - replace original target_label region with watershed result
            max_existing_label = self.tiffMask.max()
            unique_regions = np.unique(ws_labels)
            unique_regions = unique_regions[unique_regions > 0]  # Exclude background

            for i, region_id in enumerate(unique_regions):
                region_mask = (ws_labels == region_id)
                new_label = max_existing_label + i + 1
                self.tiffMask[region_mask] = new_label

            # Clear original target_label (replaced by new labels)
            self.tiffMask[target_region & (ws_labels == 0)] = 0

            # Refresh UI
            self.actions.saveMask.setEnabled(True)
            self.updateUniqueLabelListFromEntireMask()
            self.loadAnnotationsAndMasks()
            self.openNextImg(nextN=0)  # Refresh current slice display
            
            # Clear seed points
            self.canvas.clearWatershedSeeds()
            
            # Show optimization effect information
            volume_reduction = ((z_max-z_min+1) * (y_max-y_min+1) * (x_max-x_min+1)) / (self.tiffMask.shape[0] * self.tiffMask.shape[1] * self.tiffMask.shape[2])
            speedup_estimate = 1 / volume_reduction if volume_reduction > 0 else 1
            
            self.statusBar().showMessage(
                f"🚀 Optimized 3D watershed completed! "
                f"Created {len(unique_regions)} new regions. "
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

    def find_connected_slice(self):
        """
        Finds and navigates to a slice where the given label is a single large connected component.
        """
        try:
            label_to_find = int(self.find_connected_slice_input.text())
        except ValueError:
            self.statusBar().showMessage("Please enter a valid integer label.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("No mask data available.")
            return

        self.statusBar().showMessage(f"Searching for connected slice for label {label_to_find}...")

        for i in range(self.tiffMask.shape[self.currentViewAxis]):
            slice_mask = self.get_current_slice(self.tiffMask, i)
            
            if np.any(slice_mask == label_to_find):
                binary_mask = (slice_mask == label_to_find)
                # vvv Modified line vvv
                num_features = self.count_large_components(binary_mask, min_size=10)
                # ^^^ Modified line ^^^
                
                if num_features == 1:
                    self.currentSliceIndex = i
                    self.updateDisplayedSlice()
                    self.loadAnnotationsAndMasks()
                    self.statusBar().showMessage(f"Found connected slice for label {label_to_find} at index {i}.")
                    return

        self.statusBar().showMessage(f"No connected slice found for label {label_to_find}.")
    def find_connected_slice(self):
        """
        Finds and navigates to a slice where the given label is a single connected component.
        """
        try:
            label_to_find = int(self.find_connected_slice_input.text())
        except ValueError:
            self.statusBar().showMessage("Please enter a valid integer label.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("No mask data available.")
            return

        self.statusBar().showMessage(f"Searching for connected slice for label {label_to_find}...")

        for i in range(self.tiffMask.shape[self.currentViewAxis]):
            slice_mask = self.get_current_slice(self.tiffMask, i)
            
            # Check if the label exists on this slice
            if np.any(slice_mask == label_to_find):
                # Isolate the label and find connected components
                binary_mask = (slice_mask == label_to_find)
                _, num_features = cc3d.connected_components(binary_mask, return_N=True)
                
                if num_features == 1:
                    self.currentSliceIndex = i
                    self.updateDisplayedSlice()
                    self.loadAnnotationsAndMasks()
                    self.statusBar().showMessage(f"Found connected slice for label {label_to_find} at index {i}.")
                    return

        self.statusBar().showMessage(f"No connected slice found for label {label_to_find}.")
    
    def find_prev_connected_slice(self):
        try:
            label_to_find = int(self.find_connected_slice_input.text())
        except ValueError:
            self.statusBar().showMessage("Please enter a valid integer label.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("No mask data available.")
            return

        self.statusBar().showMessage(f"Searching for previous connected slice for label {label_to_find}...")
        
        max_slice = self.tiffMask.shape[self.currentViewAxis]
        
        # Search backward from the current slice
        for i in range(self.currentSliceIndex - 1, -1, -1):
            slice_mask = self.get_current_slice(self.tiffMask, i)
            if np.any(slice_mask == label_to_find):
                binary_mask = (slice_mask == label_to_find)
                num_features = self.count_large_components(binary_mask, min_size=10)
                if num_features == 1:
                    self.currentSliceIndex = i
                    self.updateDisplayedSlice()
                    self.loadAnnotationsAndMasks()
                    self.statusBar().showMessage(f"Found previous connected slice for label {label_to_find} at index {i}.")
                    return
        
        self.statusBar().showMessage(f"No previous connected slice found for label {label_to_find}.")


    def find_next_connected_slice(self):
        try:
            label_to_find = int(self.find_connected_slice_input.text())
        except ValueError:
            self.statusBar().showMessage("Please enter a valid integer label.")
            return

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.statusBar().showMessage("No mask data available.")
            return

        self.statusBar().showMessage(f"Searching for next connected slice for label {label_to_find}...")

        max_slice = self.tiffMask.shape[self.currentViewAxis]

        # Search forward from the current slice
        for i in range(self.currentSliceIndex + 1, max_slice):
            slice_mask = self.get_current_slice(self.tiffMask, i)
            if np.any(slice_mask == label_to_find):
                binary_mask = (slice_mask == label_to_find)
                num_features = self.count_large_components(binary_mask, min_size=10)
                if num_features == 1:
                    self.currentSliceIndex = i
                    self.updateDisplayedSlice()
                    self.loadAnnotationsAndMasks()
                    self.statusBar().showMessage(f"Found next connected slice for label {label_to_find} at index {i}.")
                    return
        
        self.statusBar().showMessage(f"No next connected slice found for label {label_to_find}.")


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

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

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
        self._update_undo_actions()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1 or self.tiffData is not None:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

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
        self.actions.saveMask.setEnabled(True)
        self.updateUniqueLabelListFromEntireMask()
        self.openNextImg(nextN=0)  # Refresh current view
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
