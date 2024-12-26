import vtk

def create_coordinate_grid(renderer):
    """
    Add a coordinate grid to the 3D scene using vtkCubeAxesActor.

    Parameters:
        renderer (vtk.vtkRenderer): The renderer where the grid will be added.
    """
    # 设置坐标网格的范围
    bounds = [-10, 10, -10, 10, -10, 10]  # x, y, z 的范围

    # 创建 CubeAxesActor
    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(bounds)
    axes.SetCamera(renderer.GetActiveCamera())  # 绑定渲染器的相机

    # 设置坐标轴的标签和网格线可见性
    axes.GetXAxesLinesProperty().SetColor(1, 0, 0)  # x 轴线的颜色（红色）
    axes.GetYAxesLinesProperty().SetColor(0, 1, 0)  # y 轴线的颜色（绿色）
    axes.GetZAxesLinesProperty().SetColor(0, 0, 1)  # z 轴线的颜色（蓝色）
    axes.GetTitleTextProperty(0).SetColor(1, 1, 1)  # x 轴标题的颜色
    axes.GetLabelTextProperty(0).SetColor(1, 1, 1)  # x 轴标签的颜色

    # 设置坐标轴标题
    axes.SetXTitle("X Axis")
    axes.SetYTitle("Y Axis")
    axes.SetZTitle("Z Axis")

    # 显示网格线
    axes.DrawXGridlinesOn()
    axes.DrawYGridlinesOn()
    axes.DrawZGridlinesOn()

    # 添加到渲染器
    renderer.AddActor(axes)

def create_sphere():
    """
    Create a sphere for visualization.
    """
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(5)
    sphere_source.SetCenter(0, 0, 0)
    sphere_source.SetThetaResolution(30)
    sphere_source.SetPhiResolution(30)
    sphere_source.Update()

    # 创建 Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere_source.GetOutputPort())

    # 创建 Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 设置球体颜色为灰色

    return actor

# 设置渲染器、窗口和交互器
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 添加球体到渲染器
sphere_actor = create_sphere()
renderer.AddActor(sphere_actor)

# 添加坐标网格到渲染器
create_coordinate_grid(renderer)

# 设置背景颜色
renderer.SetBackground(0.1, 0.2, 0.4)  # 深蓝色背景

# 调整相机视野
renderer.ResetCamera()

# 启动渲染和交互
render_window.Render()
interactor.Start()