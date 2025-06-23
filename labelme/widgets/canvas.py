import imgviz
import numpy as np
import scipy.ndimage
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt

import labelme.ai
import labelme.utils
from labelme import QT5
from labelme.logger import logger
from labelme.shape import Shape

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0

class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal(list)
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    mouseMoved = QtCore.Signal(QtCore.QPointF)
    pointSelected = QtCore.Signal(QtCore.QPointF)  # Send the selected point

    CREATE, EDIT = 0, 1
    _createMode = "polygon"
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError("Unexpected value for double_click event: {}".format(self.double_click))
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "erase": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
                "ai_boundary": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.currentSliceIdx = -1
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._ai_model = None
        self.embedding_dir = None
        
        # ---- 笔刷相关 ----
        self.brush_size = 5               # 笔刷半径(你自己也可理解为直径/2)
        self.drawing_mask = None          # 用于绘制笔刷预览的透明图层
        self.last_brush_point = None      # 上次笔刷坐标(图像坐标)

    def setBrushSize(self, size):
        self.brush_size = size

    def resetBrushPath(self):
        """不再使用 path 来保存笔画，改为在 drawing_mask 里画完就可清空"""
        self.drawing_mask = None

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
            "ai_boundary",
            'erase',
            'brush',
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def initializeAiModel(self, name, embedding_dir=None):
        if name not in [model.name for model in labelme.ai.MODELS]:
            raise ValueError("Unsupported ai model: %s" % name)
        model = [model for model in labelme.ai.MODELS if model.name == name][0]

        if self._ai_model is not None and self._ai_model.name == model.name and self.embedding_dir == embedding_dir:
            logger.debug("AI model is already initialized: %r" % model.name)
        else:
            logger.debug("Initializing AI model: %r" % model.name)
            self._ai_model = model(embedding_dir=embedding_dir)
            self.embedding_dir = embedding_dir

        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        return len(self.shapesBackups) >= 2

    def restoreShape(self):
        if not self.isShapeRestorable:
            print("Cannot restore shape")
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        print(f"Restored {len(self.shapes)} shapes")
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            self.repaint()  # clear crosshair
        else:
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)
        self.prevMovePoint = pos
        self.restoreCursor()
        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # --- 1) 笔刷模式下的移动绘制 ---
        if self.createMode == "brush" and ev.buttons() & QtCore.Qt.LeftButton:
            self.overrideCursor(CURSOR_DRAW)
            
            if self.drawing_mask is None and self.pixmap:
                self.drawing_mask = QtGui.QImage(
                    self.pixmap.size(),
                    QtGui.QImage.Format_ARGB32
                )
                self.drawing_mask.fill(QtCore.Qt.transparent)
                self.last_brush_point = pos
                
            if self.drawing_mask:
                # 在 drawing_mask(与原图同大小)上画线段
                painter = QtGui.QPainter(self.drawing_mask)
                painter.setPen(QtGui.QPen(
                    QtGui.QColor(255, 0, 0, 128),
                    self.brush_size * 2,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin
                ))
                painter.drawLine(self.last_brush_point, pos)
                painter.end()
                self.last_brush_point = pos
                self.repaint()
            return

        # --- 2) 多边形、矩形等普通绘制 ---
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode
            self.overrideCursor(CURSOR_DRAW)

            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode in ["rectangle", "erase"]:
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # --- 3) 移动/编辑形状 ---
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click & Drag to move point\nALT + SHIFT + Click to delete point")
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("ALT + Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(self.tr("Click & drag to move shape '%s'") % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # --- 1) 笔刷模式：点击开始绘制 ---
        if ev.button() == QtCore.Qt.LeftButton and self.createMode == "brush":
            # 初始化时机放在 mouseMoveEvent 一致处理也可以，这里仅示意
            if self.pixmap and self.drawing_mask is None:
                self.drawing_mask = QtGui.QImage(
                    self.pixmap.size(),
                    QtGui.QImage.Format_ARGB32
                )
                self.drawing_mask.fill(QtCore.Qt.transparent)
            self.last_brush_point = pos
            self.update()
            return

        # --- 2) 其他模式 ---
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line", "erase"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]
                        else self.createMode
                    )
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]
                        and ev.modifiers() & QtCore.Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge() and ev.modifiers() == QtCore.Qt.AltModifier:
                    self.addPointToEdge()
                elif self.selectedVertex() and ev.modifiers() == (
                    QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier
                ):
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
                self.pointSelected.emit(pos)

        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            # --- 笔刷模式结束时，生成 mask shape ---
            if self.createMode == "brush" and self.drawing_mask:
                try:
                    arr = labelme.utils.img_qt_to_arr(self.drawing_mask)
                    mask = (arr[:, :, 3] > 128).astype(np.uint8)  # 用 alpha 通道做阈值
                    y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
                    
                    mask_shape = Shape(shape_type="mask")
                    mask_shape.setShapeRefined(
                        shape_type="mask",
                        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                        point_labels=[1, 1],
                        mask=mask[y1 : y2 + 1, x1 : x2 + 1],
                    )
                    self.shapes.append(mask_shape)
                    self.storeShapes()
                    
                    # 清理临时资源
                    self.drawing_mask = None
                    self.newShape.emit([])
                    self.current = None
                    self.update()
                except Exception as e:
                    logger.error("Error finalizing brush stroke: %s", str(e))
                return
            
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )
        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()
            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return
        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask", "ai_boundary"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        if self.selectedVertex():
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # --- 1) 先把画布切换到图像坐标系(缩放+平移)并画原图 ---
        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(0, 0, self.pixmap)

        # 如果是 brush 模式，并且当前有临时的笔刷绘制图层，就叠加上去
        if self.createMode == "brush" and self.drawing_mask is not None:
            p.drawImage(0, 0, self.drawing_mask)

        # --- 2) 恢复到不缩放的设备坐标，用和原先一样的逻辑去画 shape ---
        p.scale(1 / self.scale, 1 / self.scale)

        # 画辅助十字线
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 255, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)

        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        # 实时显示 AI 的轮廓或 mask
        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent, forcing alpha=64."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)
        elif self.createMode == "ai_polygon" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            # AI 推理
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage()),
                slice_index=self.currentSliceIdx
            )
            points = self._ai_model.predict_polygon_from_points(
                points=[[pt.x(), pt.y()] for pt in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            if len(points) > 2:
                drawing_shape.setShapeRefined(
                    shape_type="polygon",
                    points=[QtCore.QPointF(pt[0], pt[1]) for pt in points],
                    point_labels=[1] * len(points),
                )
                drawing_shape.fill = self.fillDrawing()
                drawing_shape.selected = True
                drawing_shape.paint(p)
        elif self.createMode == "ai_mask" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage()),
                slice_index=self.currentSliceIdx
            )
            mask = self._ai_model.predict_mask_from_points(
                points=[[pt.x(), pt.y()] for pt in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            drawing_shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1 : y2 + 1, x1 : x2 + 1],
            )
            drawing_shape.selected = True
            drawing_shape.paint(p)
        
        elif self.createMode == "ai_boundary" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage()),
                slice_index=self.currentSliceIdx
            )
            mask = self._ai_model.predict_mask_from_points(
                points=[[pt.x(), pt.y()] for pt in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )

            # 如果AI模型返回了有效的掩码，则计算边界并显示预览
            if mask.any():
                # --- 使用与 finalise 方法中相同的边界计算逻辑 ---
                eroded_mask = scipy.ndimage.binary_erosion(mask)
                dilated_mask = scipy.ndimage.binary_dilation(mask)
                boundary_mask = dilated_mask ^ eroded_mask

                rows, cols = np.where(boundary_mask)
                if rows.size > 0:
                    y1, x1 = rows.min(), cols.min()
                    y2, x2 = rows.max(), cols.max()

                    cropped_boundary_mask = boundary_mask[y1 : y2 + 1, x1 : x2 + 1]

                    # 将预览形状设置为 'mask' 类型
                    drawing_shape.setShapeRefined(
                        shape_type="mask",
                        points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                        point_labels=[1, 1],
                        mask=cropped_boundary_mask,
                    )
                    drawing_shape.fill = self.fillDrawing()
                    drawing_shape.selected = True
                    drawing_shape.paint(p)       
        
        elif self.createMode == "rectangle" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage()),
                slice_index=self.currentSliceIdx
            )
            # 这里若需要实时显示，可在此 AI 推理
            # ...

        p.end()

    def transformPos(self, point):
        """把鼠标事件坐标(设备坐标) -> 转成图像坐标(未缩放的像素坐标)。"""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        prompt_points = []
        if self.createMode == "ai_polygon":
            assert self.current.shape_type == "points"
            points = self._ai_model.predict_polygon_from_points(
                points=[[pt.x(), pt.y()] for pt in self.current.points],
                point_labels=self.current.point_labels,
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(pt[0], pt[1]) for pt in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
            )
        elif self.createMode == "ai_mask":
            assert self.current.shape_type == "points"
            prompt_points = [[pt.x(), pt.y()] for pt in self.current.points]
            mask = self._ai_model.predict_mask_from_points(
                points=prompt_points,
                point_labels=self.current.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1 : y2 + 1, x1 : x2 + 1],
            )
        
        elif self.createMode == "ai_boundary":
            assert self.current.shape_type == "points"
            prompt_points = [[pt.x(), pt.y()] for pt in self.current.points]
            mask = self._ai_model.predict_mask_from_points(
                points=prompt_points,
                point_labels=self.current.point_labels,
            )
            # 检查原始掩码是否为空
            if not mask.any():
                self.current = None
                return

            # 1. 对原始掩码进行腐蚀和膨胀操作
            eroded_mask = scipy.ndimage.binary_erosion(mask)
            dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=4)

            # 2. 通过异或(XOR)操作得到一个2像素宽的边界
            # (dilated_mask 中有而 eroded_mask 中没有的部分)
            boundary_mask = dilated_mask ^ eroded_mask

            # 3. 获取边界掩码的边界框（bounding box），以优化存储
            rows, cols = np.where(boundary_mask)
            if rows.size == 0:
                self.current = None
                return
            y1, x1 = rows.min(), cols.min()
            y2, x2 = rows.max(), cols.max()

            # 4. 根据边界框裁剪掩码
            cropped_boundary_mask = boundary_mask[y1 : y2 + 1, x1 : x2 + 1]

            # 5. 将最终形状设置为 'mask' 类型
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=cropped_boundary_mask,
            )

        elif self.createMode == "erase":
            p1 = self.current.points[0]
            p2 = self.current.points[1]

            x1 = int(min(p1.x(), p2.x()))
            y1 = int(min(p1.y(), p2.y()))
            x2 = int(max(p1.x(), p2.x()))
            y2 = int(max(p1.y(), p2.y()))

            w = x2 - x1
            h = y2 - y1
            mask = np.ones((h, w), dtype=np.uint8)
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask,
            )
        elif self.createMode == "rectangle":
            # 假设 self.current.points[0], self.current.points[1] 就是用户绘制矩形的两个对角点
            p1 = self.current.points[0]
            p2 = self.current.points[1]

            # 分别取 x、y 的 min 和 max，确保 (x1, y1) 是左上角，(x2, y2) 是右下角
            x1 = int(min(p1.x(), p2.x()))
            y1 = int(min(p1.y(), p2.y()))
            x2 = int(max(p1.x(), p2.x()))
            y2 = int(max(p1.y(), p2.y()))

            # 拿这个合法的 box 去做后续处理:
            # 1) 传给 AI 模型得到 mask
            box_points = [[x1, y1], [x2, y2]]
            mask = self._ai_model.predict_mask_from_box(points=box_points)

            # 2) 再从 mask 里求出真实的前景 bbox(或你可以直接沿用 box_points 作为最终形状)
            yA, xA, yB, xB = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)

            # 3) 设置到 self.current
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(xA, yA), QtCore.QPointF(xB, yB)],
                point_labels=[1, 1],
                mask=mask[yA : yB + 1, xA : xB + 1],
            )
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.setHiding(False)
        self.newShape.emit(prompt_points)
        self.current = None
        self.update()

    def closeEnough(self, p1, p2):
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                zoom_factor = 1.1 if delta.y() > 0 else 0.9
                self.scale *= zoom_factor
                self.scale = max(0.1, min(10.0, self.scale))
                self.zoomRequest.emit(delta.y(), ev.pos())
                self.update()
            else:
                if self.parent().parent():
                    self.parent().parent().wheelEvent(ev)
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    zoom_factor = 1.1 if ev.delta() > 0 else 0.9
                    self.scale *= zoom_factor
                    self.scale = max(0.1, min(10.0, self.scale))
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                    self.update()
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()
                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True, slice_id=-1):
        self.pixmap = pixmap
        self.currentSliceIdx = slice_id
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()
