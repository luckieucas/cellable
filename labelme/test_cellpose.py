import tifffile as tiff 
import numpy as np
from cellpose import models,io
from tqdm import tqdm
import torch



def predict(img):
    
    model = models.Cellpose(gpu=False,  model_type='cyto3')
    masks_pred, flows, styles, diams = model.eval([img], diameter=0, channels=[0,0],
                                              niter=200) # using more iterations for bacteria    return masks
    return masks_pred
if __name__ == "__main__":
    tiff_path = "/Users/apple/Documents/postdoc/Project/nuclei/dataset_3d_chunks/slice2_worm1_code0_fov0_chunk.tif"
    tiff_image = tiff.imread(tiff_path)

    masks = np.zeros_like(tiff_image)
    for i, slice_image in enumerate(tqdm(tiff_image)):
        print(f"Predicting for slice {i}")
        mask = predict(slice_image)
        masks[i] = mask[0]
    
    tiff.imwrite(tiff_path.replace(".tif", "_cellpose.tif"), masks)