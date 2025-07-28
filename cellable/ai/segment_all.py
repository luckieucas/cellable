from cellpose import models,io

class CellPose():
    name = "cellpose"
    def __init__(self):
        self. model = models.Cellpose(gpu=False, model_type='cyto3')
        print(f"Cellpose model loaded")
                                              

    def predict(self, img):
        masks_pred, flows, styles, diams = self.model.eval(
            [img], diameter=0, channels=[0,0],niter=300
        ) # using more iterations for bacteria   
        print(f"Cellpose prediction done") 
        return masks_pred[0]


class nnUNet():
    name = "nnUNet"
    def __init__(self):
        pass

    def predict(self, img):
        pass