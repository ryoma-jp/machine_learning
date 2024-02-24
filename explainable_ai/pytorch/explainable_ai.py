
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from explainable_ai.pytorch.grad_cam import GradCAM
from explainable_ai.pytorch.utils.image import show_cam_on_image

def ExplainableAI(method, model, target_layer, input_tensor, input_images, image_names=None, output_dir=None):
    """Explainable AI method for PyTorch models.
    
    Args:
        - method (str): The explainable AI method to use. Currently, only "grad_cam" is supported.
        - model (torch.nn.Module): The PyTorch model to explain.
        - target_layer (torch.nn.Module): The target layer of the model to explain.
        - input_tensor (torch.Tensor): The input tensor to the model.
        - input_images (np.ndarray): The input image that is normalized to the model.
        - image_names (list): The names of the input images are used to save output image names. If None, the name of output images are continuous numbers.
        - output_dir (str): The directory to save the output images. If None, the images are not saved.
    """
    
    # --- Define the explainable AI method ---
    methods = {
        "grad_cam": GradCAM,
    }
    cam_func = methods[method]
    
    # --- Decide the output image names ---
    if (image_names is None):
        image_names = [f'{i}.png' for i in range(len(input_images))]
    
    # --- Run the explainable AI method ---
    with cam_func(model=model, target_layers=target_layer) as cam:
        cam_output = cam(input_tensor=input_tensor, targets=None)
        
        cam_images = []
        for i, image_name in zip(range(len(cam_output)), image_names):
            cam_image = show_cam_on_image(input_images[i], cam_output[i], use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cam_images.append(cam_image)
            
            if (output_dir is not None):
                Path(output_dir, 'cam', image_name).parent.mkdir(parents=True, exist_ok=True)
                output_path = f"{output_dir}/cam/{image_name}"
                cv2.imwrite(output_path, cam_image)
                
                Path(output_dir, 'input_image', image_name).parent.mkdir(parents=True, exist_ok=True)
                output_path = f"{output_dir}/input_image/{image_name}"
                input_image = cv2.cvtColor((input_images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, input_image)

    return cam_output, cam_images
