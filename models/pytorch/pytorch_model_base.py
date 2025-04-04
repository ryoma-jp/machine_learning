
import torch
import onnx
import tvm
from tvm import relay
from pathlib import Path

class PyTorchModelBase():
    def __init__(self, device, input_size, output_dir='outputs', pth_path=None) -> None:
        '''Initialize PyTorchModelBase
        '''

    def train(self, trainloader, epochs=10, optim_params=None, output_dir=None) -> None:
        '''Train the model
        '''
        pass

    def predict(self, testloader, score_th=0.5, save_dir=None):
        '''Predict
        '''
        pass

    def decode_predictions(self, predictions):
        '''Decode predictions
        '''
        pass
    
    def evaluate(self, testloader, score_th=0.5, save_dir=None):
        '''Evaluate
        '''
        pass

    def convert_to_onnx(self, output_file='model.onnx', input_names=['modelInput'], output_names=['modelOutput']) -> None:
        self.net.cpu().eval()
        print(f'input_shape: {self.input_size}')
        dummy_input = torch.randn(self.input_size, requires_grad=False)
        torch.onnx.export(
            self.net,
            dummy_input,
            output_file,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            verbose=False)
        
        onnx_model = onnx.load(output_file)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_model, output_file)
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

    def convert_to_arm_via_tvm_byoc(self, output_dir='arm_byoc', input_name='modelInput', output_name='modelOutput') -> None:
        '''
        Convert the PyTorch model to Arm Compute Library format via TVM using the BYOC (Bring Your Own Codegen) approach
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
        
        # --- Partition the model for Arm Compute Library ---
        #  - https://discuss.tvm.apache.org/t/byoc-how-can-i-use-arm-compute-library-integration-flow-for-qnn/8882
        #  - https://tvm.apache.org/docs/how_to/deploy/arm_compute_lib.html
        from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
        mod = partition_for_arm_compute_lib(mod)
        print(mod.astext())
        
        # --- Compile Relay model with TVM ---
        with tvm.transform.PassContext(opt_level=3):
            target = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon'
            lib = relay.build(mod, target=target, params=params)

        # --- Export the compiled model to the Arm Compute Library format ---
        lib_path = Path(output_dir, 'model.so')
        cross_compile = 'aarch64-linux-gnu-gcc'
        lib.export_library(lib_path, cc=cross_compile)
