import importlib


def _load_cellpose_modules():
    try:
        cp_models = importlib.import_module("cellpose.models")
        cp_io = importlib.import_module("cellpose.io")
        return cp_models, cp_io
    except Exception:
        return None, None


models, io = _load_cellpose_modules()
HAS_CELLPOSE = models is not None

class CellPose():
    name = "cellpose"
    available = HAS_CELLPOSE
    def __init__(self):
        if HAS_CELLPOSE:
            self.model = models.Cellpose(gpu=False, model_type='cyto3')
        else:
            self.model = None
            print("CellPose model not available - cellpose not installed")
        print(f"Cellpose model loaded")
                                              

    def predict(self, img):
        if self.model is None:
            raise RuntimeError("CellPose is unavailable in this build.")
        masks_pred, flows, styles, diams = self.model.eval(
            [img], diameter=0, channels=[0,0],niter=300
        ) # using more iterations for bacteria   
        print(f"Cellpose prediction done") 
        return masks_pred[0]


class nnUNet():
    name = "nnUNet"
    available = False
    def __init__(self):
        pass

    def predict(self, img):
        pass
