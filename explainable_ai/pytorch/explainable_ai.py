
import cv2

from explainable_ai.pytorch.grad_cam import GradCAM
from explainable_ai.pytorch.utils.image import show_cam_on_image

def ExplainableAI(method, model, target_layer, input_tensor, input_images):
    """Explainable AI method for PyTorch models.
    
    Args:
        - method (str): The explainable AI method to use. Currently, only "grad_cam" is supported.
        - model (torch.nn.Module): The PyTorch model to explain.
        - target_layer (torch.nn.Module): The target layer of the model to explain.
        - input_tensor (torch.Tensor): The input tensor to the model.
        - input_images (np.ndarray): The input image that is normalized to the model.
    """
    
    # --- Define the explainable AI method ---
    methods = {
        "grad_cam": GradCAM,
    }
    cam_func = methods[method]
    
    # --- Run the explainable AI method ---
    with cam_func(model=model, target_layers=target_layer) as cam:
        cam_output = cam(input_tensor=input_tensor, targets=None)
        
        cam_images = []
        for i in range(len(cam_output)):
            cam_image = show_cam_on_image(input_images[i], cam_output[i], use_rgb=True)
            cam_images.append(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    return cam_output, cam_images
