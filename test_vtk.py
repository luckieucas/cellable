import vtk
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile


def numpy_to_vtk_image(mask_array: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """
    Convert a 3D numpy array to vtkImageData.

    Parameters:
        mask_array (np.ndarray): 3D numpy array representing the mask.
        spacing (tuple): Spacing for the VTK image data (x, y, z).

    Returns:
        vtk.vtkImageData: Converted VTK image data.
    """
    # Ensure numpy array is in C-contiguous order
    mask_array = np.ascontiguousarray(mask_array)

    # Get dimensions of the numpy array
    depth, height, width = mask_array.shape

    # Create vtkImageData object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, depth)  # Note: (x, y, z)
    vtk_image.SetSpacing(spacing)  # Set spacing
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  # Use unsigned char for scalars

    # Copy data from numpy array to vtkImageData
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                vtk_image.SetScalarComponentFromDouble(x, y, z, 0, mask_array[z, y, x])

    return vtk_image


def create_renderer_with_numpy(mask_array, spacing=(1.0, 1.0, 1.0), smooth_sigma=1.0):
    """
    Create a VTK renderer from a 3D numpy mask array.

    Parameters:
        mask_array (np.ndarray): Input 3D numpy array.
        spacing (tuple): Spacing for the mask (x, y, z).
        smooth_sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        vtk.vtkRenderer: Configured VTK renderer.
    """
    # Apply Gaussian smoothing to the numpy array using scipy
    smoothed_array = gaussian_filter(mask_array, sigma=smooth_sigma)
    #smoothed_array = mask_array
    # Convert smoothed numpy array to vtkImageData
    print(f"unique labels mask_array: {np.unique(mask_array)}")
    print(f"unique labels smoothed_array: {np.unique(smoothed_array)}")
    vtk_image = numpy_to_vtk_image(smoothed_array, spacing=spacing)

    # Compute contours using vtkDiscreteMarchingCubes
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_image)
    contour.ComputeNormalsOn()

    # Extract contours for each unique label
    unique_labels = np.unique(mask_array)
    for idx, label in enumerate(unique_labels):
        contour.SetValue(idx, label)  # Set the contour value for each label

    # Create a lookup table to map labels to colors
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(len(unique_labels))  # Number of unique labels
    lookup_table.Build()

    # Assign a unique color to each label
    colors = [
        (0.0, 0.0, 0.0),  # Black
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.5),  # Gray
    ]

    for i, label in enumerate(unique_labels):
        r, g, b = colors[i % len(colors)]  # Cycle through predefined colors
        lookup_table.SetTableValue(i, r, g, b, 1.0)  # Set RGBA for the label

    # Create a mapper and enable scalar visibility
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.SetLookupTable(lookup_table)
    mapper.SetScalarRange(0, len(unique_labels) - 1)  # Set scalar range
    mapper.ScalarVisibilityOn()

    # Create an actor for the contour data
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground([1.0, 1.0, 1.0])  # White background
    renderer.AddActor(actor)

    return renderer


if __name__ == '__main__':
    # Generate a sample 3D numpy array with multiple labels
    mask_array = np.zeros((100, 100, 100), dtype=np.uint8)
    mask_array[10:20, 10:20, 10:20] = 1 # Label 1
    # mask_array[20:30, 20:30, 20:30] = 2  # Label 2
    # mask_array[30:40, 30:40, 30:40] = 3  # Label 3
    mask_array = tifffile.imread("/Users/apple/Documents/postdoc/Project/mito/3dem_labeled/label/c_elegans_anno5_mito.tif")

    # Define spacing for the 3D mask
    spacing = (1.0, 1.0, 1)  # (x, y, z) spacing

    # Create a renderer
    renderer = create_renderer_with_numpy(
        mask_array=mask_array,
        spacing=spacing,
        smooth_sigma=1.0,  # Standard deviation for Gaussian smoothing
    )

    # Set up the rendering window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(512, 512)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Start rendering
    render_window.Render()
    interactor.Initialize()
    interactor.Start()