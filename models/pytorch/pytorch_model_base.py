
import torch

class PyTorchModelBase():
    def convert_to_onnx(self, output_file='model.onnx', input_names=['modelInput'], output_names=['modelOutput']) -> None:
        self.net.cpu().eval()
        dummy_input = torch.randn(self.input_size, requires_grad=True)
        torch.onnx.export(
            self.net,
            dummy_input,
            output_file,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            verbose=True)
        self.net.to(self.device)
