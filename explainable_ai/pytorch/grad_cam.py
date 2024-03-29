import numpy as np
from .base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 output_dir=None,
                 reshape_transform=None,
                 calibloader=None):
        super().__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
