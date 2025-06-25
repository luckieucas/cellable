import tifffile as tiff
from labelme.ai import EfficientSamVitS,SegmentAnythingModelVitH
import labelme.ai
import os
import queue
import argparse
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage import label
import matplotlib.pyplot as plt
from tqdm import tqdm

OFFSET_LABEL = 1000



tiff_path = "/Users/apple/Documents/postdoc/Project/mito/CVEM_1k_cubes/IN7gVe4r71le6sTf1UG3/IN7gVe4r71le6sTf1UG3_496_631_em.tif"


def find_optimal_clusters(points, intensities, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow Method and Silhouette Score.

    Args:
        points (np.ndarray): An array of shape (N, 2) representing the coordinates (x, y).
        intensities (np.ndarray): A 1D array of shape (N,) representing the intensity values.
        max_clusters (int): The maximum number of clusters to test.

    Returns:
        int: The optimal number of clusters.
    """
    features = np.hstack((points, intensities.reshape(-1, 1)))
    silhouette_scores = []
    inertia_values = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))
    
    # Combine both criteria (example: prioritize silhouette score)
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range starts from 2
    
    return optimal_clusters, silhouette_scores, inertia_values

def cluster_coordinates_by_intensity_auto(points, intensities, max_clusters=10):
    """
    Automatically cluster coordinates and compute the mean of each cluster.

    Args:
        points (np.ndarray): An array of shape (N, 2) representing the coordinates (x, y).
        intensities (np.ndarray): A 1D array of shape (N,) representing the intensity values.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        dict: A dictionary with cluster labels as keys and the mean coordinates of each cluster as values.
    """
    optimal_clusters, _, _ = find_optimal_clusters(points, intensities, max_clusters)
    features = np.hstack((points, intensities.reshape(-1, 1)))
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Calculate the mean coordinates for each cluster
    cluster_means = {}
    for cluster in range(optimal_clusters):
        cluster_points = points[labels == cluster]
        cluster_means[cluster] = cluster_points.mean(axis=0)
    
    return cluster_means


def get_random_points_from_mask(mask, label, num_points):
    """
    Get random coordinates from a specific label region in the mask.

    Args:
        mask (np.ndarray): The input mask, a 2D array where each pixel has a label.
        label (int): The specific label to filter the region.
        num_points (int): The number of random points to sample.

    Returns:
        np.ndarray: An array of shape (num_points, 2) containing the coordinates (row, col).
    """
    # Get the coordinates of all pixels belonging to the specified label
    coords = np.column_stack(np.where(mask == label))
    
    # If fewer points are available than requested, return all
    if len(coords) <= num_points:
        return coords
    
    # Randomly sample the desired number of points
    sampled_indices = np.random.choice(len(coords), size=num_points, replace=False)
    return coords[sampled_indices]


def initializeAiModel(model_name):
    if model_name not in [model.name for model in labelme.ai.MODELS]:
        raise ValueError("Unsupported ai model: %s" % model_name)
    model = [model for model in labelme.ai.MODELS if model.name == model_name][0]


    sam_model = model()

    return sam_model

def compute_tiff_sam_feature(tiff_image, model_name, embedding_dir, view_axis, task_queue, stop_event):
    """
    在后台计算 TIFF 图像所有切片的 SAM embedding 特征。
    此函数现在从一个队列中获取任务，并可以被外部事件停止。

    Args:
        task_queue (queue.Queue): 一个包含待计算切片索引的队列。
        stop_event (threading.Event): 一个用于通知线程停止的事件。
    """
    sam_model = initializeAiModel(model_name)

    # 根据视图轴心来转置数据
    if view_axis == 1:
        data_to_process = np.transpose(tiff_image, (1, 0, 2))
    elif view_axis == 2:
        data_to_process = np.transpose(tiff_image, (2, 0, 1))
    else:
        data_to_process = tiff_image
    
    # 只要停止事件没有被设置，就一直处理队列中的任务
    while not stop_event.is_set():
        try:
            # 从队列中获取一个任务，设置超时以避免永久阻塞
            slice_index = task_queue.get(timeout=1)

            slice_image = data_to_process[slice_index]
            embedding_path = os.path.join(embedding_dir, f"slice_{slice_index}.npy")
            
            if os.path.exists(embedding_path):
                task_queue.task_done()
                continue
                
            print(f"Computing feature for slice {slice_index} of view axis {view_axis} in {embedding_dir}")
            sam_model.set_image(slice_image, slice_index=slice_index, embedding_dir=embedding_dir)
            
            # 标记任务完成
            task_queue.task_done()

        except queue.Empty:
            # 如果队列为空，说明所有任务都已完成，线程可以结束
            print("Embedding queue is empty. Worker thread is finishing.")
            break
        except Exception as e:
            print(f"Error during embedding computation: {e}")
            break

def relabel_mask_with_offset(labeled_mask, lbl, offset=OFFSET_LABEL):
    """
    Adjust the labeled mask such that each label > 1 is offset by 1000 times its value.

    Args:
        labeled_mask (np.ndarray): A labeled mask where connected components have unique labels.

    Returns:
        adjusted_mask (np.ndarray): The adjusted mask with new label values.
    """
    adjusted_mask = labeled_mask.copy()
    unique_labels = np.unique(labeled_mask)
    print(f"unique labels {unique_labels}")
    
    for label_value in unique_labels:
        if label_value > 0:  # Only adjust labels > 1
            adjusted_mask[labeled_mask == label_value] = lbl + (label_value-1) * offset

    return adjusted_mask.astype(np.uint16)

def correct_false_merge(image_path, mask_path):
    img = tiff.imread(image_path)
    mask = tiff.imread(mask_path)
    embedding_dir = ""
    model = initializeAiModel(model_name="EfficientSam (accuracy)")
    refined_mask = np.zeros_like(mask, dtype=np.uint16)
    for i in tqdm(range(mask.shape[0])):
        # normalize each slice of tiff data
        img_slice = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        mask_slice = mask[i]
        model.set_image(
                image=img_slice, slice_index=i, embedding_dir=embedding_dir
            )
        print(f"unique label {np.unique(mask_slice)}")
        for lbl in np.unique(mask_slice):
            if lbl == 0:
                continue
            # if lbl != 10 and lbl != 2:
            #     continue

            prompt_point_list = get_random_points_from_mask(mask_slice, lbl, 80)
            total_mask_num = np.sum(mask_slice==lbl)
            print(f"total mask num {total_mask_num}")


            mask_num_list = []
            combined_mask = np.zeros_like(mask_slice, dtype=np.uint16)
            for point in prompt_point_list:
                pred_mask = model.predict_mask_from_points(
                    points=[point],
                    point_labels=[1],
                )
                if np.sum(pred_mask) > total_mask_num*0.95:
                    combined_mask[mask_slice==lbl] += 1
                    continue
                combined_mask += pred_mask.astype(np.uint16)
                mask_num_list.append(np.sum(pred_mask))
                # if np.sum(pred_mask) < total_mask_num*0.7:
                #     print(f"False merge at slice {i}, point {point}, total mask num {np.sum(pred_mask)}")
            combined_mask[combined_mask<30] = 0
            combined_mask[combined_mask>30] = 1
            labeled_mask,_ = label(combined_mask)
            labeled_mask = relabel_mask_with_offset(labeled_mask, lbl)
            refined_mask[i]  = np.maximum(refined_mask[i], labeled_mask) # keep the largest labeled_mask
            # normalize combined mask
            #con = (combined_mask - combined_mask.min()) / (combined_mask.max() - combined_mask.min())

            refined_mask[i] = np.maximum(refined_mask[i], combined_mask)
    tiff.imwrite(mask_path.replace(".tif", "_refined.tif"), refined_mask)
if __name__ == "__main__":
    tiff_path = "/Users/apple/Documents/postdoc/Project/nuclei/vol_part0.tif"
    model_name = "EfficientSam (accuracy)"
    #compute_tiff_sam_feature(tiff_path, model_name)
    tiff_path = "/Users/apple/Documents/postdoc/Project/nuclei/dataset_3d_chunks/slice2_worm2_code2_fov5_chunk.tif"
    mask_path = "/Users/apple/Documents/postdoc/Project/nuclei/dataset_3d_chunks/slice2_worm2_code2_fov5_chunk_instanceSeg.tif"
    correct_false_merge(tiff_path, mask_path)