import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from cellable import PY2
from cellable import QT4
from cellable import __version__
from cellable import utils
from cellable.logger import logger

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
            "mask",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    mask=utils.img_b64_to_arr(s["mask"]).astype(bool)
                    if s.get("mask")
                    else None,
                    other_data={k: v for k, v in s.items() if k not in shape_keys},
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth
    
    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            print(" image data type {}".format(type(imageData)))
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)


    def save_tiff_annotations(
        self,
        filename,
        shapes,
        otherData=None,
        flags=None,
    ):
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        def convert_to_nested_slice_dict(annotation_list, filename):
            # Initialize the target dictionary
            slice_dict = {}

            # Load existing slice_dict from the file if it exists
            if osp.exists(filename):
                with open(filename, "r") as f:
                    slice_dict = json.load(f)

            # Iterate through each annotation in the list
            updated_slices = set()  # Track which slice_ids have been updated
            for annotation in annotation_list:
                slice_id = str(annotation["slice_id"])  # Get the slice ID
                shape_type = annotation["shape_type"]  # Get the shape type
                label = annotation["label"]  # Get the label
                points = annotation["points"]  # Get the points list
                mask = annotation["mask"]
                if int(label) == 0: # If the label is 0, skip the annotation
                    continue

                # If the slice_id is not in the dictionary, initialize it as an empty dictionary
                if slice_id not in slice_dict:
                    updated_slices.add(slice_id)
                    slice_dict[slice_id] = {}

                # If the shape_type is not in the slice_id, initialize it as an empty list
                if shape_type not in slice_dict[slice_id]:
                    slice_dict[slice_id][shape_type] = []

                # Extract x1, y1 (and possibly x2, y2) from points
                x1, y1 = points[0]
                if shape_type == "rectangle":
                    x2, y2 = points[1]
                    slice_dict[slice_id][shape_type].append([x1, y1, x2, y2, label])
                elif shape_type == "point":
                    slice_dict[slice_id][shape_type].append([x1, y1, label, 1])
                elif shape_type == "mask":
                    x2, y2 = points[1]
                    slice_dict[slice_id][shape_type].append([x1, y1, x2, y2, mask, label])

            return slice_dict
        data = convert_to_nested_slice_dict(shapes, filename)
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)


    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
