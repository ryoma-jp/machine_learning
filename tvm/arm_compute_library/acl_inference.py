
import tvm
import numpy as np
from pathlib import Path

def main():
    dev = tvm.cpu(0)
    lib_file = 'arm_compute_library/model.so'
    loaded_lib = tvm.runtime.load_module(lib_file)
    gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
    
    data_shape = (1, 3, 32, 32)
    data_type = np.float32
    d_data = np.random.uniform(0, 1, data_shape).astype(data_type)
    map_inputs = {'data': d_data}
    gen_module.set_input(**map_inputs)
    gen_module.run()
    output = gen_module.get_output(0)
    print(output)
    
if __name__ == '__main__':
    main()
