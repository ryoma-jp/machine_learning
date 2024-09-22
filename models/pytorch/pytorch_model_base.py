
import torch
import tvm
from tvm import relay
from pathlib import Path

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
            verbose=False)
        self.net.to(self.device)
    
    def convert_to_arm_compute_lib_via_tvm(self, output_dir='arm_compute_library', input_name='modelInput', output_name='modelOutput') -> None:
        '''
        Convert the PyTorch model to Arm Compute Library format via TVM
        '''
        # --- Create the output directory if it does not exist ---
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- Convert PyTorch model to TVM Relay ---
        input_shape = self.input_size
        input_name = 'modelInput'
        input_dtype = 'float32'
        shape_list = [(input_name, input_shape)]
        
        # --- Ensure the model is in evaluation mode ---
        self.net.eval()
        # --- Ensure the model is on the CPU ---
        self.net.cpu()

        # --- Trace the model with a dummy input ---
        dummy_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(self.net, dummy_input).eval()
        self.net.to(self.device)
        
        # --- Convert the traced model to TVM Relay ---
        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        # --- Compile Relay model with TVM ---
        #  - https://tvm.apache.org/docs/how_to/deploy/arm_compute_lib.html
        with tvm.transform.PassContext(opt_level=3):
            target = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon'
            lib = relay.build(mod, target=target, params=params)

        # --- Export the compiled model to the Arm Compute Library format ---
        lib_path = Path(output_dir, 'model.so')
        cross_compile = 'aarch64-linux-gnu-gcc'
        lib.export_library(lib_path, cc=cross_compile)
