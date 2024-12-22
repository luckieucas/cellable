"""
Functions for prompt-based segmentation with Segment Anything.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

import torch



#
# helper functions for translating mask inputs into other prompts
#


# compute the bounding box from a mask. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box_from_mask(mask, box_extension=0):
    coords = np.where(mask == 1)
    min_y, min_x = coords[0].min(), coords[1].min()
    max_y, max_x = coords[0].max(), coords[1].max()
    box = np.array([min_y, min_x, max_y + 1, max_x + 1])
    return box


# sample points from a mask. SAM expects the following point inputs:
def compute_points_from_mask(mask, original_size=None, use_single_point=True):
    box = _compute_box_from_mask(mask, box_extension=1)

    # get slice and offset in python coordinate convention
    bb = (slice(box[0], box[2]), slice(box[1], box[3]))
    offset = np.array([box[0], box[1]])

    # crop the mask and compute distances
    cropped_mask = mask[bb]

    # extend crop mask
    if use_single_point:
        cropped_mask = np.pad(cropped_mask, ((2, 2), (2, 2)), mode="constant", constant_values=0)
        offset -= 2
    object_boundaries = find_boundaries(cropped_mask, mode="outer")
    distances = gaussian(distance_transform_edt(object_boundaries == 0))
    inner_distances = distances.copy()
    cropped_mask = cropped_mask.astype("bool")
    inner_distances[~cropped_mask] = 0.0
    if use_single_point:
        center = inner_distances.argmax()
        center = np.unravel_index(center, inner_distances.shape)
        point_coords = (center + offset)[None]
        point_labels = np.ones(1, dtype="uint8")
        return point_coords[:, ::-1], point_labels

    outer_distances = distances.copy()
    outer_distances[cropped_mask] = 0.0

    # sample positives and negatives from the distance maxima
    inner_maxima = peak_local_max(inner_distances, exclude_border=False, min_distance=3)
    outer_maxima = peak_local_max(outer_distances, exclude_border=False, min_distance=5)

    # derive the positive (=inner maxima) and negative (=outer maxima) points
    point_coords = np.concatenate([inner_maxima, outer_maxima]).astype("float64")
    point_coords += offset

    if original_size is not None:
        scale_factor = np.array([
            original_size[0] / float(mask.shape[0]), original_size[1] / float(mask.shape[1])
        ])[None]
        point_coords *= scale_factor

    # get the point labels
    point_labels = np.concatenate(
        [
            np.ones(len(inner_maxima), dtype="uint8"),
            np.zeros(len(outer_maxima), dtype="uint8"),
        ]
    )
    return point_coords[:, ::-1], point_labels
#
# other helper functions
#

def _process_box(box, shape, box_extension=0):
    if box_extension == 0:  # no extension
        extension_y, extension_x = 0, 0
    elif box_extension >= 1:  # extension by a fixed factor
        extension_y, extension_x = box_extension, box_extension
    else:  # extension by fraction of the box len
        len_y, len_x = box[2] - box[0], box[3] - box[1]
        extension_y, extension_x = box_extension * len_y, box_extension * len_x

    box = np.array([
        max(box[1] - extension_x, 0), max(box[0] - extension_y, 0),
        min(box[3] + extension_x, shape[1]), min(box[2] + extension_y, shape[0]),
    ])

    # round up the bounding box values
    box = np.round(box).astype(int)

    return box

# Select the correct tile based on average of points
# and bring the points to the coordinate system of the tile.
# Discard points that are not in the tile and warn if this happens.
