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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_1" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_1/maxresdefault.jpg" width="600" alt="Launch Cellable Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_1/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 1: Launching Cellable Application (Click to watch on YouTube)</em></p>
</div>

---

### **🖥️ Interface Overview**

#### **Main Window Layout**
* **Toolbar**: File operations, AI segmentation, view adjustments
* **Canvas Area**: Displays current image or 3D slice
* **Label List**: Shows all current annotations
* **Status Bar**: Displays slice index, zoom level, current tool

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_2" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_2/maxresdefault.jpg" width="600" alt="Interface Overview Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_2/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 2: Interface Overview and Navigation (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_3" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_3/maxresdefault.jpg" width="600" alt="Load Data Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_3/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 3: Loading Different Types of Data (Click to watch on YouTube)</em></p>
</div>

---

### **️ View Navigation & Operations**

#### **Basic Operations**
* **Mouse Scroll**: Change zoom level
* **Arrow Keys/Slider**: Move between slices
* **Drag**: Pan the view

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_4" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_4/maxresdefault.jpg" width="600" alt="Navigation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_4/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 4: View Navigation Operations (Click to watch on YouTube)</em></p>
</div>

---

### **✏️ Annotation Tools**

#### **1. Polygon Tool - Manual Contour Drawing**
* Click on canvas to create vertices
* Double-click to complete drawing
* Right-click to edit vertices

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_5" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_5/maxresdefault.jpg" width="600" alt="Polygon Tool Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_5/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 5: Using the Polygon Tool (Click to watch on YouTube)</em></p>
</div>

#### **2. Mask Tool - Region Painting**
* Select brush size
* Paint mask regions
* Use eraser to remove areas

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_6" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_6/maxresdefault.jpg" width="600" alt="Mask Tool Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_6/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 6: Using the Mask Tool (Click to watch on YouTube)</em></p>
</div>

---

### **🤖 AI-Assisted Segmentation**

#### **SAM (Segment Anything Model) Segmentation**
1. Select the AI tool
2. Click inside the region of interest
3. Automatic segmentation generation

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_7" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_7/maxresdefault.jpg" width="600" alt="SAM AI Segmentation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_7/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 7: SAM AI Segmentation Demo (Click to watch on YouTube)</em></p>
</div>

#### **Efficient SAM - Fast Segmentation**
* Faster segmentation speed
* Suitable for batch processing

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_8" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_8/maxresdefault.jpg" width="600" alt="Efficient SAM Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_8/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 8: Efficient SAM Fast Segmentation (Click to watch on YouTube)</em></p>
</div>

#### **Text-to-Annotation Conversion**
* Input descriptive text
* Automatic annotation generation

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_9" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_9/maxresdefault.jpg" width="600" alt="Text to Annotation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_9/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 9: Text-to-Annotation Conversion (Click to watch on YouTube)</em></p>
</div>

---

### **🔧 Mask Editing & Optimization**

#### **Shape Editing**
* Move, resize, or delete shapes
* Merge or split regions
* Adjust brightness/contrast

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_10" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_10/maxresdefault.jpg" width="600" alt="Mask Editing Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_10/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 10: Mask Editing Operations (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_11" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_11/maxresdefault.jpg" width="600" alt="Watershed Segmentation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/watch?v=Xt_3Pjgxnl8" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 11: Watershed Segmentation for Instance Separation (Click to watch on YouTube)</em></p>
</div>

---

### **🎨 3D Rendering & Visualization**

#### **VTK 3D Viewer**
* **View → 3D Viewer**
* VTK-based 3D visualization of masks
* Rotate, zoom, and inspect segmented structures

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_12" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_12/maxresdefault.jpg" width="600" alt="3D Viewer Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_12/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 12: Using the 3D Viewer (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_13" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_13/maxresdefault.jpg" width="600" alt="Save Export Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_13/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 13: Save and Export Operations (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_14" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_14/maxresdefault.jpg" width="600" alt="Batch Processing Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_14/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 14: Batch Processing Features (Click to watch on YouTube)</em></p>
</div>

#### **Annotation Quality Control**
* Overlap detection
* Completeness checking
* Statistical reports

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_15" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_15/maxresdefault.jpg" width="600" alt="Quality Control Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_15/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 15: Annotation Quality Control (Click to watch on YouTube)</em></p>
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

## **🚀 YouTube上传方案**

### **1. 更新README使用YouTube链接**

```markdown:README.md
<code_block_to_apply_changes_from>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_1" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_1/maxresdefault.jpg" width="600" alt="Launch Cellable Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_1/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 1: Launching Cellable Application (Click to watch on YouTube)</em></p>
</div>

---

### **🖥️ Interface Overview**

#### **Main Window Layout**
* **Toolbar**: File operations, AI segmentation, view adjustments
* **Canvas Area**: Displays current image or 3D slice
* **Label List**: Shows all current annotations
* **Status Bar**: Displays slice index, zoom level, current tool

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_2" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_2/maxresdefault.jpg" width="600" alt="Interface Overview Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_2/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 2: Interface Overview and Navigation (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_3" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_3/maxresdefault.jpg" width="600" alt="Load Data Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_3/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 3: Loading Different Types of Data (Click to watch on YouTube)</em></p>
</div>

---

### **️ View Navigation & Operations**

#### **Basic Operations**
* **Mouse Scroll**: Change zoom level
* **Arrow Keys/Slider**: Move between slices
* **Drag**: Pan the view

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_4" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_4/maxresdefault.jpg" width="600" alt="Navigation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_4/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 4: View Navigation Operations (Click to watch on YouTube)</em></p>
</div>

---

### **✏️ Annotation Tools**

#### **1. Polygon Tool - Manual Contour Drawing**
* Click on canvas to create vertices
* Double-click to complete drawing
* Right-click to edit vertices

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_5" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_5/maxresdefault.jpg" width="600" alt="Polygon Tool Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_5/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 5: Using the Polygon Tool (Click to watch on YouTube)</em></p>
</div>

#### **2. Mask Tool - Region Painting**
* Select brush size
* Paint mask regions
* Use eraser to remove areas

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_6" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_6/maxresdefault.jpg" width="600" alt="Mask Tool Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_6/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 6: Using the Mask Tool (Click to watch on YouTube)</em></p>
</div>

---

### **🤖 AI-Assisted Segmentation**

#### **SAM (Segment Anything Model) Segmentation**
1. Select the AI tool
2. Click inside the region of interest
3. Automatic segmentation generation

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_7" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_7/maxresdefault.jpg" width="600" alt="SAM AI Segmentation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_7/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 7: SAM AI Segmentation Demo (Click to watch on YouTube)</em></p>
</div>

#### **Efficient SAM - Fast Segmentation**
* Faster segmentation speed
* Suitable for batch processing

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_8" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_8/maxresdefault.jpg" width="600" alt="Efficient SAM Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_8/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 8: Efficient SAM Fast Segmentation (Click to watch on YouTube)</em></p>
</div>

#### **Text-to-Annotation Conversion**
* Input descriptive text
* Automatic annotation generation

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_9" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_9/maxresdefault.jpg" width="600" alt="Text to Annotation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_9/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 9: Text-to-Annotation Conversion (Click to watch on YouTube)</em></p>
</div>

---

### **🔧 Mask Editing & Optimization**

#### **Shape Editing**
* Move, resize, or delete shapes
* Merge or split regions
* Adjust brightness/contrast

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_10" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_10/maxresdefault.jpg" width="600" alt="Mask Editing Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_10/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 10: Mask Editing Operations (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_11" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_11/maxresdefault.jpg" width="600" alt="Watershed Segmentation Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_11/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 11: Watershed Segmentation for Instance Separation (Click to watch on YouTube)</em></p>
</div>

---

### **🎨 3D Rendering & Visualization**

#### **VTK 3D Viewer**
* **View → 3D Viewer**
* VTK-based 3D visualization of masks
* Rotate, zoom, and inspect segmented structures

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_12" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_12/maxresdefault.jpg" width="600" alt="3D Viewer Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_12/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 12: Using the 3D Viewer (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_13" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_13/maxresdefault.jpg" width="600" alt="Save Export Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_13/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 13: Save and Export Operations (Click to watch on YouTube)</em></p>
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
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_14" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_14/maxresdefault.jpg" width="600" alt="Batch Processing Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_14/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 14: Batch Processing Features (Click to watch on YouTube)</em></p>
</div>

#### **Annotation Quality Control**
* Overlap detection
* Completeness checking
* Statistical reports

<div align="center">
  <a href="https://www.youtube.com/watch?v=VIDEO_ID_15" target="_blank">
    <img src="https://img.youtube.com/vi/VIDEO_ID_15/maxresdefault.jpg" width="600" alt="Quality Control Tutorial">
    <div style="position: relative; display: inline-block;">
      <img src="https://www.youtube.com/s/desktop/VIDEO_ID_15/img/favicon_144.png" 
           style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px;">
    </div>
  </a>
  <p><em>Video 15: Annotation Quality Control (Click to watch on YouTube)</em></p>
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

### **2. 创建YouTube频道和播放列表**

#### **A. 频道设置建议**
- **频道名称**: "Cellable Tutorials" 或 "Cell Organelle Labeling"
- **频道描述**: 专注于电子显微镜图像标注和AI分割教程
- **频道图标**: 使用Cellable的logo

#### **B. 播放列表组织**
```
📚 Cellable Tutorial Series
├──  Getting Started (1-4)
├── ✏️ Annotation Tools (5-6) 
├──  AI Features (7-9)
├──  Advanced Tools (10-12)
└──  Export & Quality (13-15)
```

### **3. 视频制作脚本**

```python
# youtube_tutorial_creator.py
import os
import json
from datetime import datetime

class YouTubeTutorialCreator:
    def __init__(self):
        self.tutorials = {
            "01_open_file": {
                "title": "Cellable Tutorial 1: Launching and Opening Files",
                "description": "Learn how to launch Cellable and open your first image or TIFF stack for annotation.",
                "tags": ["cellable", "tutorial", "image annotation", "electron microscopy", "python"],
                "category": "Science & Technology",
                "thumbnail": "01_open_file_thumb.jpg"
            },
            "02_interface_overview": {
                "title": "Cellable Tutorial 2: Interface Overview and Navigation",
                "description": "Complete tour of the Cellable interface including toolbar, canvas, and label list.",
                "tags": ["cellable", "interface", "tutorial", "software", "annotation tools"],
                "category": "Science & Technology",
                "thumbnail": "02_interface_thumb.jpg"
            },
            "03_load_data": {
                "title": "Cellable Tutorial 3: Loading Different Data Types",
                "description": "How to load PNG, JPG, and multi-page TIFF files for 2D and 3D annotation.",
                "tags": ["cellable", "data loading", "tiff", "3d data", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "03_load_data_thumb.jpg"
            },
            "04_navigation": {
                "title": "Cellable Tutorial 4: View Navigation and Zoom Controls",
                "description": "Master zoom, pan, and slice navigation in Cellable for efficient annotation workflow.",
                "tags": ["cellable", "navigation", "zoom", "pan", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "04_navigation_thumb.jpg"
            },
            "05_polygon_tool": {
                "title": "Cellable Tutorial 5: Polygon Tool for Manual Annotation",
                "description": "Step-by-step guide to using the polygon tool for precise manual contour drawing.",
                "tags": ["cellable", "polygon tool", "manual annotation", "contour drawing", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "05_polygon_thumb.jpg"
            },
            "06_mask_tool": {
                "title": "Cellable Tutorial 6: Mask Tool for Region Painting",
                "description": "Learn to use the mask tool for painting and editing annotation regions.",
                "tags": ["cellable", "mask tool", "painting", "region editing", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "06_mask_thumb.jpg"
            },
            "07_ai_sam": {
                "title": "Cellable Tutorial 7: AI-Assisted Segmentation with SAM",
                "description": "Harness the power of Segment Anything Model (SAM) for automatic cell segmentation.",
                "tags": ["cellable", "AI", "SAM", "segmentation", "machine learning", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "07_sam_thumb.jpg"
            },
            "08_efficient_sam": {
                "title": "Cellable Tutorial 8: Fast Segmentation with Efficient SAM",
                "description": "Use Efficient SAM for rapid segmentation when speed is crucial.",
                "tags": ["cellable", "efficient sam", "fast segmentation", "AI", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "08_efficient_thumb.jpg"
            },
            "09_text_to_annotation": {
                "title": "Cellable Tutorial 9: Text-to-Annotation Conversion",
                "description": "Convert text descriptions into annotations using AI-powered text understanding.",
                "tags": ["cellable", "text to annotation", "AI", "natural language", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "09_text_thumb.jpg"
            },
            "10_mask_editing": {
                "title": "Cellable Tutorial 10: Advanced Mask Editing and Optimization",
                "description": "Master advanced techniques for editing, merging, and refining annotation masks.",
                "tags": ["cellable", "mask editing", "advanced techniques", "optimization", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "10_editing_thumb.jpg"
            },
            "11_watershed_segmentation": {
                "title": "Cellable Tutorial 11: Watershed Segmentation for Instance Separation",
                "description": "Learn to separate touching cell instances using watershed segmentation algorithms.",
                "tags": ["cellable", "watershed", "instance separation", "cell biology", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "11_watershed_thumb.jpg"
            },
            "12_3d_viewer": {
                "title": "Cellable Tutorial 12: 3D Visualization with VTK Viewer",
                "description": "Explore your 3D annotations using the powerful VTK-based 3D viewer.",
                "tags": ["cellable", "3D viewer", "VTK", "visualization", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "12_3d_thumb.jpg"
            },
            "13_save_export": {
                "title": "Cellable Tutorial 13: Saving and Exporting Annotations",
                "description": "Save your work in multiple formats including JSON, VOC, and COCO.",
                "tags": ["cellable", "save", "export", "formats", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "13_export_thumb.jpg"
            },
            "14_batch_processing": {
                "title": "Cellable Tutorial 14: Batch Processing Multiple Files",
                "description": "Efficiently annotate multiple images using batch processing features.",
                "tags": ["cellable", "batch processing", "efficiency", "workflow", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "14_batch_thumb.jpg"
            },
            "15_quality_control": {
                "title": "Cellable Tutorial 15: Quality Control and Validation",
                "description": "Ensure annotation quality with overlap detection and completeness checking.",
                "tags": ["cellable", "quality control", "validation", "best practices", "tutorial"],
                "category": "Science & Technology",
                "thumbnail": "15_quality_thumb.jpg"
            }
        }
    
    def generate_upload_script(self):
        """生成YouTube上传脚本"""
        script = """# YouTube Upload Script for Cellable Tutorials
# 使用 youtube-upload 库批量上传

# 安装依赖
# pip install youtube-upload

import os
import subprocess
from datetime import datetime

def upload_tutorial(video_file, title, description, tags, category, thumbnail):
    \"\"\"上传单个教程视频\"\"\"
    cmd = [
        'youtube-upload',
        '--title', title,
        '--description', description,
        '--tags', ','.join(tags),
        '--category', category,
        '--thumbnail', thumbnail,
        '--privacy', 'public',
        video_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully uploaded: {title}")
            return True
        else:
            print(f"❌ Failed to upload: {title}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception uploading {title}: {e}")
        return False

def main():
    \"\"\"主上传函数\"\"\"
    tutorials = {
"""
        
        for key, info in self.tutorials.items():
            script += f"""        "{key}": {{
            "title": "{info['title']}",
            "description": "{info['description']}",
            "tags": {info['tags']},
            "category": "{info['category']}",
            "thumbnail": "{info['thumbnail']}"
        }},\n"""
        
        script += """    }
    
    # 上传所有教程
    for key, info in tutorials.items():
        video_file = f"tutorials/videos/{key}.mp4"
        if os.path.exists(video_file):
            print(f"\\n📤 Uploading: {key}")
            upload_tutorial(
                video_file,
                info["title"],
                info["description"],
                info["tags"],
                info["category"],
                info["thumbnail"]
            )
        else:
            print(f"⚠️  Video file not found: {video_file}")
    
    print("\\n🎉 Upload process completed!")

if __name__ == "__main__":
    main()
"""
        
        with open("youtube_upload.py", "w", encoding="utf-8") as f:
            f.write(script)
        
        print("✅ Generated YouTube upload script: youtube_upload.py")
    
    def generate_playlist_description(self):
        """生成播放列表描述"""
        description = """🔬 Cellable 3D Segmentation Tutorial Series

Complete guide to using Cellable for cell organelle labeling and annotation in electron microscopy images.

 Tutorial Series:
1. Launching and Opening Files
2. Interface Overview and Navigation  
3. Loading Different Data Types
4. View Navigation and Zoom Controls
5. Polygon Tool for Manual Annotation
6. Mask Tool for Region Painting
7. AI-Assisted Segmentation with SAM
8. Fast Segmentation with Efficient SAM
9. Text-to-Annotation Conversion
10. Advanced Mask Editing and Optimization
11. Watershed Segmentation for Instance Separation
12. 3D Visualization with VTK Viewer
13. Saving and Exporting Annotations
14. Batch Processing Multiple Files
15. Quality Control and Validation

🛠️ Features Covered:
• 2D/3D Image Annotation
• AI-Powered Segmentation
• Volume Data Analysis
• 3D Visualization
• Multiple Export Formats
• Quality Control Tools

🔗 Resources:
• GitHub: https://github.com/luckieucas/cellable
• Documentation: [Add your docs link]
• Community: [Add your community link]

#Cellable #ImageAnnotation #ElectronMicroscopy #AI #Segmentation #Tutorial #Python #Science

---
Created with ❤️ for the scientific community
Last updated: {date}
""".format(date=datetime.now().strftime("%B %Y"))
        
        with open("playlist_description.txt", "w", encoding="utf-8") as f:
            f.write(description)
        
        print("✅ Generated playlist description: playlist_description.txt")
    
    def generate_thumbnail_templates(self):
        """生成缩略图模板说明"""
        template = """# Thumbnail Template Specifications

## 尺寸要求
- 推荐尺寸: 1280 x 720 像素 (16:9 比例)
- 最小尺寸: 640 x 360 像素
- 文件格式: JPG 或 PNG
- 文件大小: < 2MB

## 设计元素
1. 标题: "Cellable Tutorial [编号]"
2. 副标题: 简短功能描述
3. 软件截图: 显示相关功能
4. 品牌元素: Cellable logo
5. 视觉提示: 箭头、高亮、图标

## 命名规范
- 01_open_file_thumb.jpg
- 02_interface_thumb.jpg
- 03_load_data_thumb.jpg
... 等等

## 设计工具推荐
- Canva (免费在线设计工具)
- GIMP (免费图像编辑)
- Photoshop (专业工具)
- Figma (协作设计)

## 缩略图制作步骤
1. 创建 1280x720 画布
2. 添加渐变背景
3. 插入软件截图
4. 添加标题和描述
5. 插入品牌元素
6. 添加视觉提示
7. 导出为 JPG 格式
"""
        
        with open("thumbnail_guide.txt", "w", encoding="utf-8") as f:
            f.write(template)
        
        print("✅ Generated thumbnail guide: thumbnail_guide.txt")

# 使用示例
if __name__ == "__main__":
    creator = YouTubeTutorialCreator()
    creator.generate_upload_script()
    creator.generate_playlist_description()
    creator.generate_thumbnail_templates()
```

### **4. 视频制作最佳实践**

#### **A. 视频规格**
- **分辨率**: 1920x1080 (1080p) 或 1280x720 (720p)
- **帧率**: 30 FPS
- **时长**: 3-8分钟 (保持观众注意力)
- **格式**: MP4 (H.264编码)

#### **B. 内容结构**
```
📹 视频开头 (0-10秒)
├── 标题展示
├── 软件logo
└── 简短介绍

 主要内容 (10秒-结束前30秒)
├── 功能演示
├── 操作步骤
├── 实际案例
└── 技巧分享

🎯 视频结尾 (最后30秒)
├── 总结要点
├── 下一步建议
└── 订阅和点赞提醒
```

#### **C. 录制技巧**
- 使用清晰的鼠标指针
- 添加操作提示和字幕
- 保持稳定的录制速度
- 避免背景噪音

### **5. 上传和SEO优化**

#### **A. 标题优化**
- 包含关键词: "Cellable", "Tutorial", "Image Annotation"
- 使用数字: "Tutorial 1", "Part 1"
- 描述具体功能: "Opening Files", "AI Segmentation"

#### **B. 描述优化**
- 前3行包含关键信息
- 添加时间戳链接
- 包含相关链接和标签
- 使用表情符号增加可读性

#### **C. 标签策略**
- 主要标签: cellable, tutorial, image annotation
- 功能标签: AI, segmentation, electron microscopy
- 技术标签: python, software, science
- 长尾标签: cell organelle labeling, EM image analysis

### **6. 社区建设**

#### **A. 互动策略**
- 回复所有评论
- 创建社区帖子
- 举办问答直播
- 分享用户案例

#### **B. 跨平台推广**
- GitHub README链接
- 学术会议展示
- 社交媒体分享
- 邮件列表推广

这样您就可以创建一个专业的YouTube教程系列，为Cellable用户提供高质量的学习资源，同时提升项目的知名度和影响力。

