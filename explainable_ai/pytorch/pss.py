
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

from .parameter_space_saliency.utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
from .parameter_space_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency

class ParameterSpaceSaliency():
    def __init__(self, model, target_layers=None, output_dir='output'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.output_dir = output_dir
        if (not Path(output_dir).exists()):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        layer_to_filter_id = {}
        ind = 0
        for layer_num, (name, param) in enumerate(model.named_parameters()):
            # print(name, param.shape)
            if len(param.size()) == 4:
                # if 'conv' not in name:
                #     print('Not a conv layer {}: {}'.format(name, layer_num))
                for j in range(param.size()[0]):
                    if name not in layer_to_filter_id:
                        layer_to_filter_id[name] = [ind + j]
                    else:
                        layer_to_filter_id[name].append(ind + j)

                ind += param.size()[0]
        
        total = 0
        for layer in layer_to_filter_id:
            total += len(layer_to_filter_id[layer])
        print('Total filters:', total)
        print('Total layers:', len(layer_to_filter_id))
        
    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List,
                 targets_names: List):
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(self.model, input_tensor, targets, 'filter_wise')
        
        filter_stats_file = Path(self.output_dir, 'filter_stats.pth')
        torch.save({'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad}, filter_stats_file)
        
        inv_transform_test = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        grads_to_save, filter_saliency = self.compute_input_space_saliency(
                                            input_tensor.to(self.device), torch.tensor(targets), self.model,
                                            filter_testset_mean_abs_grad, filter_testset_std_abs_grad, 
                                            inv_transform_test, targets_names)
        gradients_heatmap = self.save_gradients(grads_to_save, input_tensor, inv_transform_test)
        
        return gradients_heatmap
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
        
    def save_gradients(self, grads_to_save, reference_image, inv_transform_test):
        input_num, input_ch, input_h, input_w = reference_image.shape
        grads_to_save, _ = grads_to_save.max(dim=1)
        grads_to_save = grads_to_save.detach().cpu().numpy().reshape((input_num, input_h, input_w))
        grads_to_save = np.abs(grads_to_save)
        # grads_to_save[grads_to_save < 0] = 0.0

        #Percentile thresholding
        grads_to_save[grads_to_save > np.percentile(grads_to_save, 99)] = np.percentile(grads_to_save, 99)
        grads_to_save[grads_to_save < np.percentile(grads_to_save, 90)] = np.percentile(grads_to_save, 90)

        grads_to_save = (grads_to_save - np.min(grads_to_save)) / (np.max(grads_to_save) - np.min(grads_to_save))

        #Superimpose gradient heatmap
        gradients_heatmap = np.ones_like(grads_to_save) - grads_to_save
        gradients_heatmap = cv2.GaussianBlur(gradients_heatmap, (3, 3), 0)

        return gradients_heatmap

    def compute_input_space_saliency(self, reference_inputs, reference_targets, net, 
                                    testset_mean_stat=None, testset_std_stat=None, inv_transform_test = None,
                                    readable_labels = None):
        #First, log things
        with torch.no_grad():
            ref_image_to_log = inv_transform_test(reference_inputs[0].detach().cpu()).permute(1, 2, 0)


            reference_outputs = net(reference_inputs)
            _, reference_predicted = reference_outputs.max(1)
            
            # Log classes:
            print("""\n
            Image target label: {}
            Image target class name: {}
            Image predicted label: {}
            Image predicted class name: {}\n
            """.format(reference_targets[0].item(),
                readable_labels[reference_targets[0].item()],
                reference_predicted[0].item(),
                readable_labels[reference_predicted[0].item()]))

        #Compute filter saliency
        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=self.device, mode='std',
                                            aggregation='filter_wise', signed=False, logit=False,
                                            logit_difference=False)
        reference_inputs, reference_targets = reference_inputs.to(self.device), reference_targets.to(self.device)

        grad_samples = []
        #Errors are a fragile concept, we should not perturb too much, we will end up on the object
        for noise_iter in range(1):
            perturbed_inputs = reference_inputs.detach().clone()
            perturbed_inputs = (1-1)*perturbed_inputs + 0*torch.randn_like(perturbed_inputs)

            perturbed_outputs = net(perturbed_inputs)
            _, perturbed_predicted = perturbed_outputs.max(1)
            # print(readable_labels[int(perturbed_predicted[0])])

            #Backprop to the input
            perturbed_inputs.requires_grad_()
            #Find the true saliency:
            filter_saliency = filter_saliency_model(
                perturbed_inputs, reference_targets,
                testset_mean_abs_grad=testset_mean_stat,
                testset_std_abs_grad=testset_std_stat).to(self.device)

            #Find the top-k salient filters
            if False:
                sorted_filters = torch.randperm(filter_saliency.size(0)).cpu().numpy()
            else:
                sorted_filters = torch.argsort(filter_saliency, descending=True).cpu().numpy()

            #Boost them:
            filter_saliency_boosted = filter_saliency.detach().clone()
            filter_saliency_boosted[sorted_filters[:10]] *= 100.0

            #Form matching loss and take the gradient:
            matching_criterion = torch.nn.CosineSimilarity()
            matching_loss = matching_criterion(filter_saliency[None, :], filter_saliency_boosted[None, :])
            matching_loss.backward()

            grads_to_save = perturbed_inputs.grad.detach().cpu()
            grad_samples.append(grads_to_save)
        #Find averaged gradients (smoothgrad-like)
        grads_to_save = torch.stack(grad_samples).mean(0)

        return grads_to_save, filter_saliency


    def sort_filters_layer_wise(self, filter_profile, layer_to_filter_id, filter_std = None):
        layer_sorted_profile = []
        means = []
        stds = []
        for layer in layer_to_filter_id:
            layer_inds = layer_to_filter_id[layer]
            layer_sorted_profile.append(np.sort(filter_profile[layer_inds])[::-1])
            means.append(np.ones_like(filter_profile[layer_inds])*np.mean(filter_profile[layer_inds]))
            if filter_std is not None:
                stds.append(filter_std[layer_inds][np.argsort(filter_profile[layer_inds])[::-1]])
        layer_sorted_profile = np.concatenate(layer_sorted_profile)
        sal_means = np.concatenate(means)
        if filter_std is not None:
            sal_stds = np.concatenate(stds)
            return layer_sorted_profile, sal_means, sal_stds
        else:
            return layer_sorted_profile, sal_means
        
        