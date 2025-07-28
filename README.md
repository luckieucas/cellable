<h1 align="center">
  <img src="labelme/icons/icon.png"><br/>cellable
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
<a href="#installation"><b>Installation</b></a>

  <!-- | <a href="https://github.com/labelmeai/labelme/discussions"><b>Community</b></a> -->
  <!-- | <a href="https://www.youtube.com/playlist?list=PLI6LvFw0iflh3o33YYnVIfOpaO0hc5Dzw"><b>Youtube FAQ</b></a> -->
</div>

<br/>

<div align="center">
  <img src="examples/instance_segmentation/.readme/annotation.jpg" width="70%">
</div>




## Installation

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

## Runing
```
conda activate cellable

cellable
```



## Acknowledgement

This repo is developed from [wkentaro/labelme](https://github.com/wkentaro/labelme).
