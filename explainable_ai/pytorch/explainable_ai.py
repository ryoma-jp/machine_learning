
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from explainable_ai.pytorch.grad_cam import GradCAM
from explainable_ai.pytorch.eigen_cam import EigenCAM
from explainable_ai.pytorch.pss import ParameterSpaceSaliency
from explainable_ai.pytorch.utils.image import show_cam_on_image

class ExplainableAI():
    def __init__(self, method, model, target_layer, output_dir=None, calib_tensor=None, calib_targets=None):
        """Explainable AI method for PyTorch models.
        
        Args:
            - method (str): The explainable AI method to use. Supported methods are "grad_cam", "eigen_cam", and "pss".
            - model (torch.nn.Module): The PyTorch model to explain.
            - target_layer (torch.nn.Module): The target layer of the model to explain.
            - output_dir (str): The directory to save the output images. If None, the images are not saved.
        """
        # --- Define the explainable AI method ---
        methods = {
            "grad_cam": GradCAM,
            "eigen_cam": EigenCAM,
            "pss": ParameterSpaceSaliency,
        }
        xai_func = methods[method]
        
        self.output_dir = output_dir
        self.xai = xai_func(model=model, target_layers=target_layer, output_dir=self.output_dir, input_tensor=calib_tensor, targets=calib_targets)
        
    def __call__(self, input_images, input_tensor, targets=None, targets_names=None, image_names=None):
        """Run the explainable AI method.
        
        Args:
            - input_tensor (torch.Tensor): The input tensor to the model.
            - input_images (np.ndarray): The input image that is normalized to the model.
            - targets (list): The target categories of the input images. If None, the model's prediction is used as the target.
            - targets_names (list): The names of the target categories. If None, the target categories are continuous numbers.
            - image_names (list): The names of the input images are used to save output image names. If None, the name of output images are continuous numbers.
        """
        # --- Decide the output image names ---
        if (image_names is None):
            image_names = [f'{i}.png' for i in range(len(input_images))]
        
        xai_output = self.xai(input_tensor=input_tensor, targets=targets, targets_names=targets_names)
        
        xai_images = []
        for i, image_name in zip(range(len(xai_output)), image_names):
            xai_image = show_cam_on_image(input_images[i], xai_output[i], use_rgb=True)
            xai_image = cv2.cvtColor(xai_image, cv2.COLOR_RGB2BGR)
            xai_images.append(xai_image)
            
            if (self.output_dir is not None):
                Path(self.output_dir, 'heatmap', image_name).parent.mkdir(parents=True, exist_ok=True)
                output_path = f"{self.output_dir}/heatmap/{image_name}"
                cv2.imwrite(output_path, xai_image)
                
                Path(self.output_dir, 'input_image', image_name).parent.mkdir(parents=True, exist_ok=True)
                output_path = f"{self.output_dir}/input_image/{image_name}"
                input_image = cv2.cvtColor((input_images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, input_image)

        return xai_output, xai_images
        

