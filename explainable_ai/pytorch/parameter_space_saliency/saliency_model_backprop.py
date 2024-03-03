
import torch.autograd as autograd
import torch
import torch.nn as nn
from .utils import show_heatmap_on_image, AverageMeter
from utils.utils import SinglePassVarianceComputation
import time

class SaliencyModel(nn.Module):
    def __init__(self, net, criterion, device='cuda', mode='std', aggregation='filter_wise', signed=False, logit=False,
                 logit_difference=False):
        super(SaliencyModel, self).__init__()
        self.net = net
        self.criterion = criterion
        self.device = device
        self.mode = mode
        self.aggregation = aggregation
        self.signed = signed
        self.logit = logit
        self.logit_difference = logit_difference

    def forward(self, inputs, targets, target_to_subtract=None, testset_mean_abs_grad=None, testset_std_abs_grad=None):
        # Turn on gradient for the input image
        inputs.requires_grad_()

        # Set the net in eval mode
        self.net.eval()

        inputs, targets = inputs.to(self.device), targets.to(self.device),

        if  target_to_subtract is not None:
            target_to_subtract = target_to_subtract.to(self.device)

        self.net.zero_grad()
        outputs = self.net(inputs)

        if self.logit:
            raise NotImplementedError
            #loss = outputs[0][int(targets[0])]
        elif self.logit_difference:
            raise NotImplementedError
            #loss = outputs[0][int(targets[0])] - outputs[0][int(target_to_subtract[0])]
        else:
            loss = self.criterion(outputs, targets)


        gradients = autograd.grad(loss, self.net.parameters(), create_graph=True, allow_unused=True)

        filter_grads = []
        for i in range(len(gradients)):  # Filter-wise aggregation
            # print(gradients[i].size())

            if self.aggregation == 'filter_wise':
                if len(gradients[i].size()) == 4:  # If conv layer
                    if not self.signed:
                        # first take abs and then aggregate
                        filter_grads.append(gradients[i].abs().mean(-1).mean(-1).mean(-1))
                    else:
                        filter_grads.append(gradients[i].mean(-1).mean(-1).mean(-1))
            if self.aggregation == 'parameter_wise':
                if not self.signed:
                    filter_grads.append(gradients[i].view(-1).abs())
                else:
                    filter_grads.append(gradients[i].view(-1))
            if self.aggregation == 'tensor_wise':
                raise NotImplementedError

        if not self.signed:
            naive_saliency = torch.abs(torch.cat(filter_grads))
        else:
            naive_saliency = torch.cat(filter_grads)
        if self.mode == 'naive':
            return naive_saliency
        if self.mode == 'std':
            testset_std_abs_grad[testset_std_abs_grad <= 1e-14] = 1  # This should fix nans in the resulting parameter_saliency
            std_saliency = (naive_saliency - testset_mean_abs_grad.to(self.device)) / testset_std_abs_grad.to(
                self.device)
            return std_saliency
        if self.mode == 'norm':
            testset_mean_abs_grad[testset_mean_abs_grad <= 1e-14] = 1
            norm_saliency = naive_saliency / testset_mean_abs_grad.to(self.device)
            return norm_saliency

def find_testset_saliency(net, calibloader, aggregation):
    """find_saliency is a basic parameter_saliency method: could be naive, could be averaging across filters, tensors, layers, etc
    Return average magnitude of gradient across samples in the testset and std of that"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Go through images one by one, cannot go in batches since we need avg abs grad, not just avg grad
    # Incrementally compute the mean and std to not run out of memory
    iter_time = AverageMeter()
    end = time.time()
    
    #input_tensor, targets = next(iter(calibloader))
    #testset_inputs, testset_targets = input_tensor.to(device), targets.to(device)
    #testset_outputs = net(testset_inputs)
    #_, testset_predicted = testset_outputs.max(1)

    filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=device, mode='naive',
                                            aggregation=aggregation, signed=False, logit=False,
                                            logit_difference=False)
    spvc = SinglePassVarianceComputation()
    for input_tensor, targets in calibloader:
        testset_inputs, testset_targets = input_tensor.to(device), targets.to(device)
        testset_grad = filter_saliency_model(testset_inputs, testset_targets).detach().to(device)
        
        testset_mean_abs_grad, testset_std_abs_grad = spvc(testset_grad)
#    testset_grad = filter_saliency_model(testset_inputs, testset_targets).detach().to(device)
#
#    # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
#    testset_mean_abs_grad_prev = torch.zeros_like(testset_grad, dtype=torch.float64)
#    testset_mean_abs_grad = testset_grad
#    # print(testset_mean_abs_grad)
#    testset_std_abs_grad = (testset_grad - testset_mean_abs_grad) * (testset_grad - testset_mean_abs_grad_prev)
#
#    testset_std_abs_grad = testset_std_abs_grad / float(len(input_tensor) - 1)  # Unbiased estimator of variance
#    print('Variance:', testset_std_abs_grad)
#    testset_std_abs_grad = torch.sqrt(testset_std_abs_grad)
#    print('Std:', testset_std_abs_grad)
#    print('Mean:', testset_mean_abs_grad)
#    print('Testset_grads_shape:{}'.format(testset_mean_abs_grad.shape))

    return testset_mean_abs_grad, testset_std_abs_grad

if __name__ == '__main__':
    pass