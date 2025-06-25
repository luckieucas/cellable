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
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt



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
from labelme.label_file import LabelFileError
from labelme.logger import logger
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.utils import compute_tiff_sam_feature, compute_points_from_mask
from PyQt5.QtWidgets import QSplitter, QRadioButton, QLineEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QWidgetAction, QLineEdit, QPushButton, QLabel,  QSizePolicy

try:
    from . import utils
except:
    import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()
OFFSET_LABEL = 1000
MAX_LABEL = 2000

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera


from PyQt5.QtCore import QThread, pyqtSignal


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
        self.rotation_speed = 0.3  # 设置旋转灵敏度，值越小旋转越慢
        self.zoom_speed = 0.5     # 设置缩放灵敏度，值越小缩放越慢

    def Rotate(self):
        # 减慢旋转速度
        self.MotionFactor *= self.rotation_speed
        super().Rotate()

    def Dolly(self):
        # 减慢缩放速度
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

    def _create_persistent_crosshair(self):
        """仅在初始化时调用，创建十字线的Actor并添加到场景中。"""
        color = (1.0, 0.0, 0.0)  # 红色
        radius = 2.0

        # 1. 创建中心球体
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

        # 2. 创建三条正交线
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

        # 初始时将它们全部设为不可见
        for actor in self.crosshair_actors:
            actor.SetVisibility(False)


    def update_crosshair_position(self, center_point, data_shape):
        """更新十字线的位置，并确保其可见。"""
        if not self.crosshair_actors: # 如果还没创建好就返回
            return
        depth, height, width = data_shape
        x, y, z = center_point

        # 更新中心球体的位置
        self._crosshair_sources['sphere'].SetCenter(x, y, z)

        # 更新三条线的位置
        self._crosshair_sources['x'].SetPoint1(0, y, z)
        self._crosshair_sources['x'].SetPoint2(width, y, z)

        self._crosshair_sources['y'].SetPoint1(x, 0, z)
        self._crosshair_sources['y'].SetPoint2(x, height, z)

        self._crosshair_sources['z'].SetPoint1(x, y, 0)
        self._crosshair_sources['z'].SetPoint2(x, y, depth)

        # 确保所有十字线 actor 都是可见的
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


    def numpy_to_vtk_image(self, data: np.ndarray):
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

        # Wrap the numpy array into a VTK array
        vtk_array = numpy_support.numpy_to_vtk(num_array=data.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        # Set the VTK array as the scalars for the vtkImageData
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image

    def add_grid(self, data: np.ndarray):
        """
        Add a coordinate grid to the 3D scene based on the input data's shape.

        Parameters:
            data (np.ndarray): 3D numpy array to determine grid bounds.
        """
        # Get the bounds from the data shape
        depth, height, width = data.shape
        bounds = [0, width, 0, height, 0, depth]  # x, y, z ranges

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

        # Add the axes actor to the renderer
        self.renderer.AddActor(axes)

    # def highlight_point_with_crosshair(self, point: tuple, data_shape: tuple, color=(1.0, 0.0, 0.0), radius=1.0):
    #     """
    #     Highlight a given point with a crosshair in the 3D rendered scene, with line limits constrained by data bounds.

    #     Parameters:
    #         point (tuple): The (x, y, z) coordinates of the point to highlight.
    #         data_shape (tuple): The shape of the data array (depth, height, width) to constrain the crosshair range.
    #         color (tuple): The RGB color of the point and crosshair (default is red).
    #         radius (float): The radius of the highlighted point (default is 1.0).
    #     """
    #     # Remove previously highlighted actors
    #     for actor in self.highlight_actors:
    #         self.renderer.RemoveActor(actor)
    #     self.highlight_actors = []  
    #     depth, height, width = data_shape

    #     # Highlight the point with a sphere
    #     sphere_source = vtk.vtkSphereSource()
    #     sphere_source.SetCenter(point)  # Set the position of the sphere
    #     sphere_source.SetRadius(radius)  # Set the size of the sphere
    #     sphere_source.SetThetaResolution(30)
    #     sphere_source.SetPhiResolution(30)

    #     mapper = vtk.vtkPolyDataMapper()
    #     mapper.SetInputConnection(sphere_source.GetOutputPort())

    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.GetProperty().SetColor(color)  # Set the color of the sphere
    #     actor.GetProperty().SetOpacity(1.0)  # Fully opaque

    #     # Add the sphere actor to the renderer
    #     self.renderer.AddActor(actor)
    #     self.highlight_actors.append(actor)


    #     # Create crosshair lines (X, Y, Z axes) within the bounds of the data
    #     for axis, (start, end) in enumerate([
    #         ((0, point[1], point[2]), (width, point[1], point[2])),  # X-axis
    #         ((point[0], 0, point[2]), (point[0], height, point[2])),  # Y-axis
    #         ((point[0], point[1], 0), (point[0], point[1], depth)),  # Z-axis
    #     ]):
    #         line_source = vtk.vtkLineSource()
    #         line_source.SetPoint1(start)
    #         line_source.SetPoint2(end)

    #         line_mapper = vtk.vtkPolyDataMapper()
    #         line_mapper.SetInputConnection(line_source.GetOutputPort())

    #         line_actor = vtk.vtkActor()
    #         line_actor.SetMapper(line_mapper)
    #         line_actor.GetProperty().SetColor(color)  # Same color as the point
    #         line_actor.GetProperty().SetLineWidth(2.0)  # Thicker line for better visibility

    #         # Set dashed line style
    #         line_actor.GetProperty().SetLineStipplePattern(0xF0F0)  # Pattern for dashed line
    #         line_actor.GetProperty().SetLineStippleRepeatFactor(3)  # Repeat factor for the dash pattern

    #         # Add the line actor to the renderer
    #         self.renderer.AddActor(line_actor)
    #         self.highlight_actors.append(line_actor)

    #     # Render the scene to update the display
    #     self.vtkWidget.GetRenderWindow().Render()

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
        # 只有在相机从未被初始化时（即第一次加载时），才重置相机
        if not self.camera_initialized:
            self.renderer.ResetCamera()
            self.camera_initialized = True  # 标记为已初始化

        # 对于后续的所有更新，我们只调用Render()，而不重置相机
        self.vtkWidget.GetRenderWindow().Render()


    def center_camera_on_point(self, point_3d):
        """
        将3D相机的焦点移动到指定的三维点，并相应地平移相机位置。
        
        :param point_3d: 一个包含 (x, y, z) 坐标的元组或列表。
        """
        # 获取当前的活动相机
        camera = self.renderer.GetActiveCamera()
        if not camera:
            return

        # 1. 获取相机当前的位置和焦点
        old_position = np.array(camera.GetPosition())
        old_focal_point = np.array(camera.GetFocalPoint())

        # 2. 我们要移动到的新焦点就是传入的3D点
        new_focal_point = np.array(point_3d)

        # 3. 计算相机相对于其焦点的偏移向量
        #    这个向量决定了您的观察角度和距离
        offset_vector = old_position - old_focal_point

        # 4. 计算相机的新位置：新的焦点 + 同样的偏移向量
        new_position = new_focal_point + offset_vector

        # 5. 设置相机的新焦点和新位置
        camera.SetFocalPoint(new_focal_point)
        camera.SetPosition(new_position)

        # 6. 重新渲染窗口以立即显示变化
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
        # ---------- 创建多根工具栏 ----------
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

        # 统一压缩样式
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

        self.labelList = LabelListWidget()
        #ßself.currentLabelList = LabelListWidget()
        self.lastOpenDir = None


        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        # Labellist for current slice of tiff data
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                rgb = self._get_rgb_by_label(label)
                item = self.uniqLabelList.createItemFromLabel(label, rgb=rgb, checked=True)
                self.uniqLabelList.addItem(item)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.uniqLabelList.itemChanged.connect(self.onUniqLabelItemChanged)

        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

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

        self.canvas = self.labelList.canvas = Canvas(
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
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        
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
        for dock in ["label_dock", "shape_dock", "file_dock"]:
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
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        
        # --- Combine label-input, Delete, and Split into a vertical layout ---
        label_ops_widget = QWidget(self)
        label_ops_widget.setFixedWidth(150)

        # Vertical container
        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(4)

        # Row 1: the label-input field
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.label_input = QLineEdit(self)
        self.label_input.setPlaceholderText("Label")
        self.label_input.setFixedWidth(40)
        input_layout.addWidget(self.label_input)
        v_layout.addLayout(input_layout)

        # Row 2: Delete Label button
        self.delete_label_button = QPushButton("Delete Label", self)
        self.delete_label_button.setFixedWidth(100)
        self.delete_label_button.clicked.connect(self.delete_label)
        v_layout.addWidget(self.delete_label_button)

        # Row 3: Split Label button
        self.split_label_button = QPushButton("Split Label", self)
        self.split_label_button.setFixedWidth(100)
        self.split_label_button.clicked.connect(self.split_label)
        v_layout.addWidget(self.split_label_button)

        label_ops_widget.setLayout(v_layout)

        # Add to toolbar as a single action
        label_ops_action = QWidgetAction(self)
        label_ops_action.setDefaultWidget(label_ops_widget)
        #self.toolbar.addAction(label_ops_action)
        # --- end vertical layout ---


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
        openNextTenImg = action(
            self.tr("&Next 10"),
            self.openNextTenImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=True,
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
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
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
            self.tr("AI-mask by box"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing Ai mask by rectangles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
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
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
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
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                embedding_dir = self.embedding_dir
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("AI-Mask by Points"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                embedding_dir = self.embedding_dir
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        createAiBoundaryMode = action(
            self.tr("AI-Boundary by Point"),
            lambda: self.toggleDrawMode(False, createMode="ai_boundary"),
            None,
            "objects",
            self.tr("Start drawing ai_boundary by points. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiBoundaryMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                embedding_dir = self.embedding_dir
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
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
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

        # vvv 在这里添加 Redo 动作 vvv
        redo = action(
            self.tr("Redo\n"),
            self.redoShapeEdit,
            shortcuts.get("redo", "Ctrl+Y"), # 假设重做快捷键为 Ctrl+Y
            "redo",
            self.tr("Redo last undone edit"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
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

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
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

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

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
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            saveMask=saveMask,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            redo=redo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            createAiBoundaryMode=createAiBoundaryMode,
            eraseMode=eraseMode,
            createBrushMode=createBrushMode,
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
            fileMenuActions=(open_, opendir, saveAs, close, quit),
            tool=(
                createAiMaskMode, 
                createRectangleMode,
                createAiBoundaryMode, 
                eraseMode, 
                createBrushMode,
                brush_action, 
                editMode,
                label_ops_action,
                merge_labels_action,
            ),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                paste,
                delete,
                None,
                undo,
                redo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createRectangleMode,
                createAiMaskMode,
                createPointMode,
                createMode,
                editMode,
                edit,
                duplicate,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

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
                saveAs,
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
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
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
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                embedding_dir=self.embedding_dir
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]
            else None
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

        # ---------- 文件 / 导航 ----------
        utils.addActions(self.file_toolbar,
            (open_, openPrevImg, openNextImg,saveMask))

        # ---------- 绘制 / 标签 ----------
        self.draw_toolbar.addActions([
            createAiMaskMode, 
            createAiBoundaryMode,
            createRectangleMode, 
            eraseMode, 
            createBrushMode,
        ])
        
        self.draw_toolbar.addAction(brush_action)
        self.draw_toolbar.addAction(label_ops_action)
        self.draw_toolbar.addAction(merge_labels_action)

        # ---------- 视图 / 其它 ----------
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
        

        
        # 1) 增加一个布尔变量，表示是否允许更新 LabelList
        self.enableUpdateLabelList = True

        # 2) 创建复选框
        self.checkBoxUpdateLabelList = QtWidgets.QCheckBox("Update LabelList")
        self.checkBoxUpdateLabelList.setChecked(False)  # 默认勾选
        self.checkBoxUpdateLabelList.stateChanged.connect(self.onUpdateLabelListCheckBoxChanged)
        self.checkBoxUpdateLabelList.setLayoutDirection(QtCore.Qt.RightToLeft)

        # 3) 放到你想放的工具栏/布局里，这里演示加到 self.toolbar
        checkBoxAction = QtWidgets.QWidgetAction(self)
        checkBoxAction.setDefaultWidget(self.checkBoxUpdateLabelList)
        self.view_toolbar.addAction(checkBoxAction)

        # initialize lastClickedPoint so it always exists
        self.lastClickedPoint = None


        # --- Add a 3D rendering mode toggle ---
        # 1) Track whether to show all labels in 3D
        self.showAll3D = True

        self.crosshair_center_xy = None

        # 2) Create the checkbox widget
        self.checkBox3DRendering = QtWidgets.QCheckBox("Show All 3D")
        self.checkBox3DRendering.setChecked(False)
        self.checkBox3DRendering.stateChanged.connect(self.on3DRenderingCheckBoxChanged)
        self.checkBox3DRendering.setLayoutDirection(QtCore.Qt.RightToLeft)

        # --- 创建一个新的组合控件来容纳 View Selection 和 3D 相关按钮 (垂直三行版) ---

        # 1. 创建最外层的“容器”小部件和它的主垂直布局
        view_3d_controls_widget = QtWidgets.QWidget()
        main_v_layout = QtWidgets.QVBoxLayout(view_3d_controls_widget)
        main_v_layout.setContentsMargins(5, 5, 5, 5)
        main_v_layout.setSpacing(2)
        main_v_layout.setAlignment(QtCore.Qt.AlignTop) # 让所有控件顶部对齐

        # 2. 创建并添加第一行：View Selection
        #    (确保 self.viewSelection 已经被创建)
        top_row_layout = QtWidgets.QHBoxLayout()
        top_row_layout.addWidget(QtWidgets.QLabel("View:"))
        top_row_layout.addWidget(self.viewSelection)
        main_v_layout.addLayout(top_row_layout)

        # 3. 直接将 CheckBox 作为第二行添加到主垂直布局
        #    (确保 self.checkBox3DRendering 已被创建)
        main_v_layout.addWidget(self.checkBox3DRendering)

        # 4. 直接将 Update 3D Button 作为第三行添加到主垂直布局
        #    (确保 self.update3DButton 已被创建)
        main_v_layout.addWidget(self.update3DButton)

        # 5. 将这个新的组合控件包装在一个 QWidgetAction 中
        view_3d_controls_action = QtWidgets.QWidgetAction(self)
        view_3d_controls_action.setDefaultWidget(view_3d_controls_widget)

        # 6. 最后，将这个新的 Action 添加到 view_toolbar
        self.view_toolbar.addAction(view_3d_controls_action)

        # --- 新的组合控件创建结束 ---
        self.label_visibility_states = {}


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

    def noShapes(self):
        return not len(self.labelList)


    def populateModeActions(self):
        # 1) 先清空 draw_toolbar 上已有的 Action
        for act in list(self.draw_toolbar.actions()):
            self.draw_toolbar.removeAction(act)

        # 2) 往 draw_toolbar 里重新添加“画图/标签”相关的工具按钮
        utils.addActions(self.draw_toolbar, self.actions.tool)

        # 3) 更新 Canvas 的右键菜单
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.actions.menu)

        # 4) 更新主窗口的 Edit 菜单
        self.menus.edit.clear()
        edit_actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.createAiBoundaryMode,
            self.actions.editMode,
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
            self.saveLabels(label_file)
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
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        self.actions.createAiBoundaryMode.setEnabled(True)
        self.actions.eraseMode.setEnabled(True)
        self.actions.createBrushMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

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
        self.labelList.clear()
        #self.currentLabelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.tiffData = None
        self.tiffJsonAnno = None # Annotation from tiff file
        self.tiffMask = None # Mask for tiffdata
        self.annotation_json = None
        self.tiff_mask_file = None
        self.labelFile = None
        self.otherData = None
        self.currentSliceIndex = -1
        self.currentAIPromptPoints = [] # current ai prompt [((x1,y1),label1),((x2,y2),label2),...]
        self.embedding_dir = None # embedding dir for efficient sam
        self.current_mask_num = 0 # current number of label in mask for current slice predicted by ai model
        self.canvas.resetState()
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.camera_initialized = False
        self.segmentAllModel = None
        self.label_list = [i for i in range(1,MAX_LABEL)] 
        self.sliceCache = {}

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        if not self.canvas.isUndoable:
            return
        self.canvas.undo()
        # 使用修复后的 loadShapes 来高效刷新UI
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
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "erase": self.actions.eraseMode,
            "brush": self.actions.createBrushMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

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

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

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

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        self._update_undo_actions()
        for item in items:
            shape: Shape = item.shape()

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            # update mask
            if shape.shape_type == "mask":
                self._update_mask_to_tiffMask(shape)

            self._update_shape_color(shape)
            if shape.group_id is None:
                item.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                    )
                )
            else:
                item.setText("{} ({})".format(shape.label, shape.group_id))
            self.setDirty()
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                rgb = self._get_rgb_by_label(shape.label)
                item = self.uniqLabelList.createItemFromLabel(shape.label, rgb, checked=True)
                self.uniqLabelList.addItem(item)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)

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

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

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
        self.labelList.addItem(label_list_item)
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

    def onUpdateLabelListCheckBoxChanged(self, state: int):
        """
        Update `enableUpdateLabelList` when the check box is checked or unchecked.
        If the checkbox was just checked after being unchecked, refresh all shapes in `canvas.shapes` to `labelList`.
        If the checkbox was just unchecked after being checked, clear `labelList`.
        """
        self.enableUpdateLabelList = (state == QtCore.Qt.Checked)

        if self.enableUpdateLabelList:
            # 重新勾选时，labelList与canvas重新同步
            self.labelList.clear()
            for shape in self.canvas.shapes:
                self.addLabel(shape)
        else:
            # 取消勾选时，清空 labelList
            self.labelList.clear()
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
            # 1) 先算出颜色
            rgb = LABEL_COLORMAP[int(label) % len(LABEL_COLORMAP)]
            # 2) 确保列表里有这个标签条目
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label, rgb=rgb, checked=True)
                self.uniqLabelList.addItem(item)
            # 3) 更新图标
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

    def remLabels(self, shapes):
        for shape in shapes:
            if shape.shape_type == "mask":
                self._remove_shape_from_mask(shape)
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)


    def addLabelMinimal(self, shape):
        """
        Perform minimal operations for shape during wheel scrolling.
        """
        self._update_shape_color(shape)  # Only update the shape color


    def addLabelComplete(self, shape):
        """
        Perform the complete addLabel operation for shape.
        """
        if not self.enableUpdateLabelList:
            return
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            rgb = self._get_rgb_by_label(shape.label)
            item = self.uniqLabelList.createItemFromLabel(shape.label, rgb, checked=True)
            self.uniqLabelList.addItem(item)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        # Update the shape color
        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )
        # 从全局状态字典中获取此标签的可见性，如果未记录则默认为 True (可见)
        is_visible = self.label_visibility_states.get(shape.label, True)
        self.canvas.setShapeVisible(shape, is_visible)

    def loadShapesFromTiff(self, shapes, replace=True):
        """
        Load shapes with optimized behavior for wheel scrolling and stopping.
        """
        self._noSelectionSlot = True

        # Call minimal operation for each shape during scrolling
        for shape in shapes:
            self.addLabelMinimal(shape)

        # Clear selection
        self.labelList.clearSelection()
        self._noSelectionSlot = False

        # Load shapes into the canvas
        self.canvas.loadShapes(shapes, replace=replace)

        # Start a timer to trigger the complete operation after scrolling stops
        self.startAddLabelCompleteTimer(shapes)
    def _update_undo_actions(self):
        self.actions.undo.setEnabled(self.canvas.isUndoable)
        self.actions.redo.setEnabled(self.canvas.isRedoable)
    
    def loadShapes(self, shapes, replace=True):
            self._noSelectionSlot = True
            if replace: # <-- 这是关键的修复
                self.labelList.clear() # 在替换模式下，先清空UI列表

            for shape in shapes:
                self.addLabel(shape)

            self.labelList.clearSelection()
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


    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    slice_id=s.slice_id,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] and self.tiffData is None else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            if self.tiffJsonAnno is not None:
                lf.save_tiff_annotations(filename=filename,shapes=shapes)
            else:
                lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)
    
    def onUniqLabelItemChanged(self, item: QtWidgets.QListWidgetItem):
        return
        label = item.data(Qt.UserRole)            # 字符串
        visible = (item.checkState() == Qt.Checked)
        
        self.label_visibility_states[label] = visible

        # 1) Canvas 中的形状可见性
        # for shape in self.canvas.shapes:
        #     if shape.label == label:
        #         self.canvas.setShapeVisible(shape, visible)

        # 2) Polygon Labels 列表里的条目同步
        #    LabelListWidget 可直接迭代，yield 的是 QListWidgetItem
        for li in self.labelList:
            if li.shape().label == label:
                li.setCheckState(Qt.Checked if visible else Qt.Unchecked)

        # 3) 3-D 视图同步（可选）
        try:
            lbl_int = int(label)
            self.vtk_widget.toggle_label_visibility(lbl_int, visible)
        except Exception:
            pass

        self.canvas.update()


    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        try:
            label = int(shape.label)
        except ValueError as e: 
            print("Label is not int")
        # Set render visible in 3d view
        if self.vtk_widget is not None:
            self.vtk_widget.toggle_label_visibility(label, item.checkState() == Qt.Checked)
    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])


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
                        self.get_current_slice(self.tiffData, pred_slice_index), slice_index=pred_slice_index
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
        根据当前视图，将2D画布坐标和切片索引转换为3D空间坐标 (X, Y, Z)。
        """
        canvas_x = canvas_pos.x()
        canvas_y = canvas_pos.y()
        slice_idx = self.currentSliceIndex

        if self.currentViewAxis == 0:  # Axial 视图 (XY平面)
            # 画布(x, y) -> 3D(X, Y), 切片 -> Z
            point_3d = (canvas_x, canvas_y, slice_idx)
        elif self.currentViewAxis == 1:  # Coronal 视图 (XZ平面)
            # 画布(x, y) -> 3D(X, Z), 切片 -> Y
            point_3d = (canvas_x, slice_idx, canvas_y)
        elif self.currentViewAxis == 2:  # Sagittal 视图 (YZ平面)
            # 画布(x, y) -> 3D(Y, Z), 切片 -> X
            point_3d = (slice_idx, canvas_x, canvas_y)
        else:
            # 默认情况或错误情况
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
            self.labelList.clearSelection()
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
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
            self._update_undo_actions()
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

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)


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
                self.tiffData = tiff.imread(filename).astype(np.uint16)
                # normalize each slice of tiff data
                for i in range(len(self.tiffData)):
                    self.tiffData[i] = self.normalizeImg(self.tiffData[i])
                print(f"TIFF data shape: {self.tiffData.shape}")
                file_dir = osp.dirname(filename)
                cell_name = osp.basename(filename).split(".")[0]
                model_name = self._selectAiModelComboBox.currentText()
                self.embedding_dir=f"{file_dir}/{cell_name}_embeddings_{model_name}_axis{self.currentViewAxis}"
                self.canvas.initializeAiModel(
                    name=self._selectAiModelComboBox.currentText(),
                    embedding_dir = self.embedding_dir
                    )
                print(f"Initialize ai model with Embedding dir: {self.embedding_dir}")
                if not os.path.exists(self.embedding_dir) or len(os.listdir(self.embedding_dir)) < len(self.tiffData):
                    # Comute features when embedding dir does not exist or not enough embeddings
                    # Start a background thread to calculate features
                    background_thread = threading.Thread(target=compute_tiff_sam_feature,  args=(self.tiffData,model_name, self.embedding_dir, self.currentViewAxis), daemon=True)
                    background_thread.start()
                self.currentSliceIndex = 0

                if self.tiffData.ndim == 3:
                    # Assuming the 3D image is a stack of 2D images, take the first slice
                    self.imageData = self.normalizeImg(self.get_current_slice(self.tiffData,0))  # Load the first slice for display
                    self.imagePath = filename
                    h, w = self.imageData.shape
                    bytes_per_line = self.imageData.strides[0]  # 对 uint8 数组而言，通常等于 w
                    self.image = QImage(
                        self.imageData.data,    # 像素缓冲区
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
        if self.tiffMask is not None:
            mask_data = self.get_current_slice(self.tiffMask, slice_index)
            for label in np.unique(mask_data):
                if label == 0:
                    continue  # Skip the background
                mask = mask_data == label
                y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
                drawing_shape = Shape(
                    label=str(label),
                    shape_type="mask",
                    description=f"Mask for label {label}",
                    slice_id=slice_index
                )
                drawing_shape.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=mask[y1 : y2 + 1, x1 : x2 + 1],
                )
                shapes.append(drawing_shape)


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
        根据选择的视图平面更新显示的切片。
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
        pixmap = QtGui.QPixmap.fromImage(image)
        self.canvas.loadPixmap(pixmap, slice_id=self.currentSliceIndex)
        
        if hasattr(self, 'tiffData') and self.tiffData is not None:
            # 如果用户从未点击过，默认将十字线放在切片中心
            if self.crosshair_center_xy is None:
                h, w = self.get_current_slice(self.tiffData).shape[:2]
                center_x, center_y = w / 2, h / 2
            else:
                center_x, center_y = self.crosshair_center_xy

            # --- 使用新的辅助函数来获取正确的3D坐标 ---
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
                self.labelList.clear()  # Clear the label list
                self.currentSliceIndex -= nextN  # Update to the previous slice index

                self.updateDisplayedSlice()
                # Delay loading annotations and masks
                QtCore.QTimer.singleShot(0, self.loadAnnotationsAndMasks)

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


    def openNextTenImg(self, _value=False):
        self.openNextImg(nextN=10)

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
                self.labelList.clear()  # Clear the label list
                self.currentSliceIndex += nextN  # Update to the next slice index
                self.updateDisplayedSlice()

                # Delay loading annotations and masks
                QtCore.QTimer.singleShot(0, self.loadAnnotationsAndMasks)

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
        根据 **整个** tiffMask 来同步 unique label list。
        这个方法会添加缺失的标签，并移除那些在遮罩中已不存在的标签，
        从而确保列表始终反映整个三维体数据的标签全集。
        """
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            self.uniqLabelList.clear() # 如果没有mask，则清空列表
            return

        # 1. 从整个3D Mask中获取所有非零的唯一标签
        #    使用集合（set）以提高后续操作的效率
        labels_in_mask = {str(l) for l in np.unique(self.tiffMask) if l != 0}

        # 2. 获取当前UI列表中的所有标签
        labels_in_widget = set()
        for i in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(i)
            labels_in_widget.add(item.data(QtCore.Qt.UserRole))

        # 3. 添加新标签：找出在Mask中存在但在UI列表中缺失的标签
        labels_to_add = labels_in_mask - labels_in_widget
        if labels_to_add:
            # 使用 natsort.natsorted 确保标签按自然顺序（如 1, 2, 10 而不是 1, 10, 2）添加
            for label in natsort.natsorted(list(labels_to_add)):
                # 这个现有的辅助函数会自动创建并添加item
                self._get_rgb_by_label(label)

        # 4. 移除旧标签：找出在UI列表中存在但已从Mask中消失的标签
        labels_to_remove = labels_in_widget - labels_in_mask
        if labels_to_remove:
            for label in labels_to_remove:
                item = self.uniqLabelList.findItemByLabel(label)
                if item:
                    # takeItem会从列表中移除指定的item
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

        # # 在将形状加载到画布前，根据全局状态字典设置每个形状的可见性
        # for shape in shapes:
        #     # 从全局状态字典中获取可见性，如果该标签没有记录，则默认为 True (可见)
        #     is_visible = self.label_visibility_states.get(shape.label, True)
        #     shape.visible = is_visible  # 直接设置 shape 对象的属性

        # Update the canvas with the loaded annotations and masks
        self.canvas.storeShapes()
        #self.loadShapes(shapes, replace=False)
        #if self.canvas.createMode != "erase":
        self.loadShapesFromTiff(shapes, replace=False)
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
            # 判断滚轮方向：向上滚动加载上一张切片，向下滚动加载下一张切片
            if event.angleDelta().y() > 0:  # 滚轮向上
                self.openPrevImg()
            else:  # 滚轮向下
                self.openNextImg()
            event.accept()
        else:
            # 如果不是 TIFF 数据，可以执行其他操作或忽略
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
        tiff.imwrite(self.tiff_mask_file, self.tiffMask)
        self.actions.saveMask.setEnabled(False)
        self.currentAIPromptPoints = []
        print(f"Updated TIFF file saved to {self.tiff_mask_file}")

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

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

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()
            # Load latest tiff file and img mask
            with open(self.annotation_json, 'r') as f:
                self.jsonAnno = json.load(f)
            #self.tiffMask = tiff.imread(self.tiff_mask_file).astype(np.uint16)

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

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
        当用户在画布上点击时触发。
        """
        self.lastClickedPoint = point

        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            return
            
        self.crosshair_center_xy = (point.x(), point.y())

        # --- 使用新的辅助函数来获取正确的3D坐标 ---
        point_3d = self._get_3d_point_from_2d(point)
        # ----------------------------------------

        # 更新3D视图中的十字线
        # 注意：self.tiffData.shape 的顺序是 (D, H, W)，对应 (Z, Y, X)
        # 而 vtk_widget 期望的坐标顺序是 (X, Y, Z)
        self.vtk_widget.update_crosshair_position(point_3d, (self.tiffData.shape[2], self.tiffData.shape[1], self.tiffData.shape[0]))

        # 如果处于单标签渲染模式，则刷新3D视图
        if not self.showAll3D:
            self.update3D()
            
        # 将3D相机焦点移动到新的点
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
        self.status(f"Tracking from current slice {self.currentSliceIndex}")
        #self._compute_center_point()
        # tracking forward
        self.predictNextNSlices(nextN=100)

        # tracking backward
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
            size_threshold = 100  # 定义尺寸阈值
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid integer label."
            )
            return

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
        
        # [新增] 4.5) 过滤掉体积小于阈值的连通域
        if cc_map.max() > 0: # 仅在找到至少一个连通域时执行过滤
            # 使用 cc3d.statistics 高效计算每个连通域的体素数量
            stats = cc3d.statistics(cc_map)
            voxel_counts = stats['voxel_counts']
            
            # 找出所有小于阈值的连通域的标签 (注意：voxel_counts[0]是背景，我们不关心)
            small_labels = [label for label, count in enumerate(voxel_counts[1:], 1) if count < size_threshold]

            if small_labels:
                # 使用 np.isin 高效地将所有小连通域的像素值置为0（背景）
                cc_map[np.isin(cc_map, small_labels)] = 0
        
        # [修改] 重新标记，确保过滤后的标签是连续的 (1, 2, 3, ...)
        # 这样可以保证后续分配新标签时不会有空缺
        final_cc_map, num_components_after_filter = cc3d.connected_components(cc_map, connectivity=26, return_N=True)

        if num_components_after_filter == 0:
            QtWidgets.QMessageBox.information(
                self,
                "No Components",
                f"No connected components larger than {size_threshold} voxels found for label {target_label}."
            )
            # 即使没有找到组件，也要确保原ROI区域被清空
            mask[roi] = 0
            self.tiffMask = mask
            self.openNextImg(nextN=0) # 刷新视图
            return

        # 5) offset new component IDs so they don't collide with existing labels
        offset = int(mask.max())
        new_mask = mask.copy()
        
        # [修改] 使用过滤和重新标记后的 final_cc_map 来更新 new_mask
        # 首先将原ROI区域清零，防止旧标签残留
        new_mask[roi] = 0
        # 然后仅在有连通域的位置赋予新标签
        new_mask[final_cc_map > 0] = offset + final_cc_map[final_cc_map > 0]

        # 6) update the in‐memory mask and enable saving
        self.tiffMask = new_mask.astype(mask.dtype)
        self.actions.saveMask.setEnabled(True)
        self.updateUniqueLabelListFromEntireMask()

        # 7) refresh the displayed slice immediately
        self.openNextImg(nextN=0)

        # 8) [修改] inform the user how many components were created *after filtering*
        QtWidgets.QMessageBox.information(
            self,
            "Split Completed",
            f"Label {target_label} was split into {num_components_after_filter} components (size >= {size_threshold})."
        )

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

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
            self._update_undo_actions()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()
        self._update_undo_actions()

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
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
    def show_interpolate_dialog(self):
        """显示插值设置对话框"""
        if not hasattr(self, 'tiffMask') or self.tiffMask is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No mask data available to interpolate.")
            return

        # 创建并显示对话框，预填充当前切片索引
        # Find boundary label 10000 start slices and end slices
        # 1. 查找所有目标标签为10000的体素坐标
        positions = np.argwhere(self.tiffMask == 10000)
        if positions.size == 0:
            QtWidgets.QMessageBox.critical(self, "Error", "No target label 10000 found.")
            return

        # 2. 根据当前视图轴心(self.currentViewAxis)动态计算起止切片
        #    positions 的列顺序是 (z, y, x)，分别对应 axis 0, 1, 2
        slice_indices_for_view = positions[:, self.currentViewAxis]
        start_slice = int(slice_indices_for_view.min())
        end_slice = int(slice_indices_for_view.max())

        # 3. 对话框的最大切片值也应根据当前视图确定
        max_slice_for_view = self.tiffData.shape[self.currentViewAxis] - 1
        
        # 4. 创建并显示对话框，预填充我们刚刚计算出的正确值
        dialog = InterpolateDialog(self, start_slice, end_slice, max_slice_for_view)
        
        # 如果用户点击 "OK"
        if dialog.exec_():
            start_slice, end_slice, target_label_str = dialog.getValues()
            
            if not target_label_str.isdigit():
                QtWidgets.QMessageBox.critical(self, "Error", "Target Label must be an integer.")
                return

            target_label = int(target_label_str)
            
            # 显示一个等待光标，因为计算可能需要一点时间
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.run_interpolation(start_slice, end_slice, target_label)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Interpolation Error", str(e))
            finally:
                # 恢复正常光标
                QtWidgets.QApplication.restoreOverrideCursor()
            if target_label == 10000:
                # 如果目标标签是10000，表示我们在填充边界标签
                # 则需要在插值后去掉边界
                self.tiffMask[self.tiffMask == target_label] = 0

    def run_interpolation(self, start_slice, end_slice, target_label):
        """执行基于距离变换的插值算法"""
        if start_slice >= end_slice:
            raise ValueError("Start Slice must be smaller than End Slice.")

        # 1. 获取起始和结束蒙版
        mask_a = (self.get_current_slice(self.tiffMask, start_slice) == target_label)
        mask_b = (self.get_current_slice(self.tiffMask, end_slice) == target_label)

        if not mask_a.any() or not mask_b.any():
            raise ValueError(f"Label {target_label} not found on both start and end slices.")

        # 2. 计算有符号距离场 (Signed Distance Transform)
        # 内部为正，外部为负
        dt_a = distance_transform_edt(mask_a) - distance_transform_edt(~mask_a)
        dt_b = distance_transform_edt(mask_b) - distance_transform_edt(~mask_b)
        
        # 3. 循环遍历中间的每一个切片并进行插值
        total_slices = end_slice - start_slice
        for i in range(1, total_slices):
            slice_index = start_slice + i
            
            # 计算当前切片的插值权重
            weight = i / total_slices
            
            # 线性插值距离场
            interp_dt = (1.0 - weight) * dt_a + weight * dt_b
            
            # 从插值后的距离场重建蒙版 (所有距离>=0的区域即为内部)
            interp_mask = interp_dt >= 0
            
            # 4. 将生成的蒙版写回到 self.tiffMask 中
            current_slice_mask = self.get_current_slice(self.tiffMask, slice_index)
            # 首先清空该区域可能存在的旧标签，然后填充新标签
            current_slice_mask[interp_mask] = target_label
            # 如果需要，也可以保留其他标签：
            # current_slice_mask[~interp_mask & (current_slice_mask == target_label)] = 0

        # 5. 刷新UI
        self.actions.saveMask.setEnabled(True)
        self.updateUniqueLabelListFromEntireMask()
        self.openNextImg(nextN=0)  # 刷新当前视图
        
        QtWidgets.QMessageBox.information(
            self, "Success", f"Successfully interpolated label {target_label} between slices {start_slice} and {end_slice}."
        )

class InterpolateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, start_slice=-1, end_slice=-1, max_slice=100):
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
        self.end_slice_spinbox.setValue(end_slice) # 默认向后10帧

        self.target_label_label = QtWidgets.QLabel("Target Label:")
        self.target_label_input = QtWidgets.QLineEdit()
        self.target_label_input.setPlaceholderText("Enter label ID to interpolate")
        self.target_label_input.setText("10000")

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
        """返回用户输入的值"""
        return (
            self.start_slice_spinbox.value(),
            self.end_slice_spinbox.value(),
            self.target_label_input.text()
        )