import os
import collections
import threading

import imgviz
import numpy as np
import onnxruntime
import skimage

from ..logger import logger
from . import _utils


class EfficientSam:
    def __init__(self, encoder_path, decoder_path):
        self._encoder_session = onnxruntime.InferenceSession(encoder_path)
        self._decoder_session = onnxruntime.InferenceSession(decoder_path)

        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

        self._thread = None

    def set_image(self, image: np.ndarray, slice_index=None, embedding_dir=None):
        with self._lock:
            self._image = image
            self._slice_index = slice_index
            self._image_embedding = self._image_embedding_cache.get(
                self._image.tobytes()
            )
        
            # Attempt to load embedding if embedding directory is specified
            if self._image_embedding is None and embedding_dir is not None and slice_index is not None:
                embedding_path = os.path.join(
                    embedding_dir, f"slice_{slice_index}.npy"
                )
                if os.path.exists(embedding_path):
                    logger.debug(f"Loading embedding for slice {slice_index} from {embedding_dir}...")
                    self._image_embedding = np.load(embedding_path)

        if self._image_embedding is None:
            self._thread = threading.Thread(
                target=self._compute_and_cache_image_embedding,
                kwargs={'embedding_dir': embedding_dir} # <--- 通过 kwargs 传递
            )
            self._thread.start()

    def _compute_and_cache_image_embedding(self, embedding_dir=None):
        with self._lock:
            logger.debug("Computing image embedding...")
            # Ensure the image has the correct number of dimensions
            if self._image.ndim == 2:  # Grayscale image
                self._image = np.stack([self._image] * 3, axis=-1)  # Convert to pseudo-RGB
            elif self._image.ndim == 3 and self._image.shape[2] in [3, 4]:  # RGB or RGBA
                self._image = self._image
            else:
                raise ValueError(
                    f"Unsupported image shape: {self._image.shape}. Must be 2D (H, W) or 3D (H, W, C)."
                )

            image = imgviz.rgba2rgb(self._image)
            batched_images = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            (self._image_embedding,) = self._encoder_session.run(
                output_names=None,
                input_feed={"batched_images": batched_images},
            )
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[self._image.tobytes()] = self._image_embedding
            logger.debug("Done computing image embedding.")

            # Save embedding to file if embedding_dir is specified
            if embedding_dir is not None and self._slice_index is not None:
                if not os.path.exists(embedding_dir):
                    os.makedirs(embedding_dir)
                embedding_path = os.path.join(
                    embedding_dir, f"slice_{self._slice_index}.npy"
                )
                np.save(embedding_path, self._image_embedding)

    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def predict_mask_from_points(self, points, point_labels):
        return _compute_mask_from_points(
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=self._get_image_embedding(),
            points=points,
            point_labels=point_labels,
        )
    
    def predict_mask_from_box(self, points):
        return _compute_mask_from_box(
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=self._get_image_embedding(),
            points=points,
        )

    def predict_polygon_from_points(self, points, point_labels):
        mask = self.predict_mask_from_points(points=points, point_labels=point_labels)
        return _utils.compute_polygon_from_mask(mask=mask)


def _compute_mask_from_points(
    decoder_session, image, image_embedding, points, point_labels
):
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array(point_labels, dtype=np.float32)
    print(f"input label: {input_label}")

    # batch_size, num_queries, num_points, 2
    batched_point_coords = input_point[None, None, :, :]
    # batch_size, num_queries, num_points
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int64),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask = masks[0, 0, 0, :, :]  # (1, 1, 3, H, W) -> (H, W)
    mask = mask > 0.0

    MIN_SIZE_RATIO = 0.05
    skimage.morphology.remove_small_objects(
        mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
    )

    if 0:
        imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
    return mask

def _compute_mask_from_box(
    decoder_session, image, image_embedding, points
):
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array([2, 3], dtype=np.float32)
    print(f"input label: {input_label}")

    # batch_size, num_queries, num_points, 2
    batched_point_coords = input_point[None, None, :, :]
    # batch_size, num_queries, num_points
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int64),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask = masks[0, 0, 0, :, :]  # (1, 1, 3, H, W) -> (H, W)
    mask = mask > 0.0

    MIN_SIZE_RATIO = 0.05
    skimage.morphology.remove_small_objects(
        mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
    )

    if 0:
        imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
    return mask