<h1 align="center">
  <img src="labelme/icons/icon.png"><br/>Cellable
</h1>

<h4 align="center">
  Cell Organelle Labeling with Python
</h4>

<div align="center">
  <a href="https://pypi.python.org/pypi/labelme"><img src="https://img.shields.io/pypi/v/labelme.svg"></a>
  <a href="https://pypi.org/project/labelme"><img src="https://img.shields.io/pypi/pyversions/labelme.svg"></a>
  <a href="https://github.com/labelmeai/labelme/actions"><img src="https://github.com/labelmeai/labelme/workflows/ci/badge.svg?branch=main&event=push"></a>
</div>

<div align="center">
<a href="#installation"><b>Installation</b></a> |
<a href="#tutorial"><b>User Tutorial</b></a> |
<a href="#features"><b>Features</b></a>
</div>

<br/>

<div align="center">
  <img src="examples/instance_segmentation/.readme/annotation.png" width="80%">
</div>

## **Overview**

This application is an extended version of [Labelme](https://github.com/wkentaro/labelme), designed for interactive **2D/3D segmentation and annotation** of electron microscopy (EM) and other scientific images.
It supports:

* Viewing and annotating **2D slices** and **3D volumes**
* Loading **TIFF stacks** for volumetric data
* Automatic **AI-assisted segmentation**
* Manual mask editing and refinement
* 3D rendering via **VTK**

---

## **Installation**

### **1. Requirements**

* **Python 3.8+**
* GPU recommended for AI-assisted segmentation
* OS: Linux, macOS, or Windows

### **2. Install Dependencies**

Key dependencies include:

* `PyQt5` – GUI framework
* `vtk` – 3D rendering
* `tifffile` – TIFF image I/O
* `cc3d` – connected component analysis
* `scikit-image`, `scipy`, `numpy` – image processing
* `imgviz` – visualization utilities

```bash
git clone https://github.com/luckieucas/cellable.git
cd cellable

# Setup conda
conda create --name cellable python=3.9
conda activate cellable

# Install dependencies
pip install -r requirements.txt

# Install cellable
pip install -e .
```

---

## **📚 User Tutorial - Cellable 3D Segmentation Edition**

### **🚀 Getting Started**

#### **Launch the Application**
```bash
conda activate cellable
cellable
```

<div align="center">
  <video width="600" controls>
    <source src="examples/tutorial/videos/01_open_file.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 1: Launching Cellable Application</em></p>
</div>

---

### **🖥️ Interface Overview**

#### **Main Window Layout**
* **Toolbar**: File operations, AI segmentation, view adjustments
* **Canvas Area**: Displays current image or 3D slice
* **Label List**: Shows all current annotations
* **Status Bar**: Displays slice index, zoom level, current tool

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/02_interface_overview.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 2: Interface Overview and Navigation</em></p>
</div>

---

### **📁 Data Loading & Supported Formats**

#### **Supported File Formats**
* **Images**: `.png`, `.jpg`, `.tif`, `.tiff`
* **Volume Data**: Multi-page TIFF stacks

#### **Loading Data Steps**
1. **Open Image/Stack**: `File → Open`
2. For 3D TIFF stacks, a slider will appear for slice navigation

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/03_load_data.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 3: Loading Different Types of Data</em></p>
</div>

---

### **️ View Navigation & Operations**

#### **Basic Operations**
* **Mouse Scroll**: Change zoom level
* **Arrow Keys/Slider**: Move between slices
* **Drag**: Pan the view

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/04_navigation.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 4: View Navigation Operations</em></p>
</div>

---

### **✏️ Annotation Tools**

#### **1. Polygon Tool - Manual Contour Drawing**
* Click on canvas to create vertices
* Double-click to complete drawing
* Right-click to edit vertices

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/05_polygon_tool.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 5: Using the Polygon Tool</em></p>
</div>

#### **2. Mask Tool - Region Painting**
* Select brush size
* Paint mask regions
* Use eraser to remove areas

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/06_mask_tool.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 6: Using the Mask Tool</em></p>
</div>

---

### **🤖 AI-Assisted Segmentation**

#### **SAM (Segment Anything Model) Segmentation**
1. Select the AI tool
2. Click inside the region of interest
3. Automatic segmentation generation

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/07_ai_sam.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 7: SAM AI Segmentation Demo</em></p>
</div>

#### **Efficient SAM - Fast Segmentation**
* Faster segmentation speed
* Suitable for batch processing

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/08_efficient_sam.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 8: Efficient SAM Fast Segmentation</em></p>
</div>

#### **Text-to-Annotation Conversion**
* Input descriptive text
* Automatic annotation generation

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/09_text_to_annotation.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 9: Text-to-Annotation Conversion</em></p>
</div>

---

### **🔧 Mask Editing & Optimization**

#### **Shape Editing**
* Move, resize, or delete shapes
* Merge or split regions
* Adjust brightness/contrast

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/10_mask_editing.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 10: Mask Editing Operations</em></p>
</div>

---

### **🌊 Watershed Segmentation - Instance Separation**

#### **Find False Merge Feature**
1. Enter the target label ID in the Label ID input field on the right
2. Navigate to a slice containing adhered instances
3. Click the waterz button
4. Automatic boundary computation and view refresh

<div align="center">
  <img src="examples/instance_segmentation/fm.png" width="80%">
</div>

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/11_watershed_segmentation.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 11: Watershed Segmentation for Instance Separation</em></p>
</div>

---

### **🎨 3D Rendering & Visualization**

#### **VTK 3D Viewer**
* **View → 3D Viewer**
* VTK-based 3D visualization of masks
* Rotate, zoom, and inspect segmented structures

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/12_3d_viewer.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 12: Using the 3D Viewer</em></p>
</div>

---

### **💾 Save & Export**

#### **Saving Annotations**
* `File → Save` stores as `.json` format
* Mask data can be exported as NumPy arrays

#### **Export Formats**
* JSON annotation files
* VOC dataset format
* COCO dataset format

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/13_save_export.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 13: Save and Export Operations</em></p>
</div>

---

### **⌨️ Keyboard Shortcuts**

| Action | Shortcut |
|--------|----------|
| Open File | `Ctrl+O` |
| Save Annotation | `Ctrl+S` |
| Zoom | `Hold Cmd + Mouse Scroll` |
| Next Slice | `D` |
| Previous Slice | `A` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |

---

### **🚀 Advanced Features**

#### **Batch Processing**
* Multiple file annotation
* Automatic progress saving

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/14_batch_processing.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 14: Batch Processing Features</em></p>
</div>

#### **Annotation Quality Control**
* Overlap detection
* Completeness checking
* Statistical reports

<div align="center">
  <video width="600" controls>
    <source src="tutorials/videos/15_quality_control.mp4" type="video/mp4">
    Your browser does not support the video tag
  </video>
  <p><em>Video 15: Annotation Quality Control</em></p>
</div>

---

## **🎯 Features**

### **Core Features**
- ✅ **2D/3D Image Annotation**
- ✅ **AI-Assisted Segmentation (SAM, Efficient SAM)**
- ✅ **Text-to-Annotation Conversion**
- ✅ **Watershed Instance Separation**
- ✅ **3D VTK Visualization**
- ✅ **Multi-format Export Support**

### **Professional Features**
- 📊 **Volume Data Analysis**
- ✏️ **Precise Mask Editing**
- 📊 **Batch Processing Support**

---

## **❓ Troubleshooting & FAQ**

### **Performance Issues**
* **Laggy performance**: Enable GPU acceleration and close unused windows
* **Memory issues**: Reduce the number of simultaneously open files

### **Technical Issues**
* **Mask misalignment**: Check voxel dimensions in TIFF metadata
* **VTK viewer not loading**: Ensure `vtk` and `PyQt5` versions are compatible

### **AI Segmentation Issues**
* **Inaccurate segmentation**: Adjust click position, use manual editing for optimization
* **Model loading failure**: Check network connection and model file integrity

---

## **🔗 Advanced Tutorials**

### **Custom Annotation Workflows**
* Create annotation templates
* Set annotation rules
* Quality check procedures

### **Data Preprocessing**
* Image enhancement
* Format conversion
* Batch renaming

---

## **🤝 Community & Support**

* **GitHub Issues**: Report bugs and feature requests
* **Discussions**: Share experiences and best practices
* **Contributing Guide**: Participate in project development

---

## **Credits**

This version builds upon the original [Labelme](https://github.com/wkentaro/labelme) and integrates:

* VTK for 3D visualization
* cc3d for connected component analysis
* AI models for auto-segmentation
* Efficient SAM for fast segmentation
* Text-to-annotation capabilities

---

## **📖 Additional Resources**

* [Original Labelme Project](https://github.com/wkentaro/labelme)
* [SAM Model Paper](https://arxiv.org/abs/2304.02643)
* [VTK Documentation](https://vtk.org/documentation/)
* [Electron Microscopy Image Processing Best Practices](https://example.com/em-best-practices)

---

<div align="center">
  <p><strong>🎉 Start using Cellable for professional cell organelle annotation!</strong></p>
  <p>For questions, check the tutorial videos or submit a GitHub Issue</p>
</div>

