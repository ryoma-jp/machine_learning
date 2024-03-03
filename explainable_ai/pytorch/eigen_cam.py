from .base_cam import BaseCAM
from .utils.svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 output_dir=None,
                 reshape_transform=None,
                 calibloader=None):
        super().__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
