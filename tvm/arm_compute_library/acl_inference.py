
import tvm
import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # --- Load the model ---
    dev = tvm.cpu(0)
    lib_file = 'arm_compute_library/model.so'
    loaded_lib = tvm.runtime.load_module(lib_file)
    gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
    
    # --- Load the input data list ---
    df_input = pd.read_csv('input_tensors/inputs.csv')
    print(df_input)
    
    # --- Inference ---
    data_shape = (1, 3, 32, 32)
    predictions = []
    processing_time = 0
    for index, row in tqdm(df_input.iterrows()):
        data = np.load(Path('input_tensors', row['input_tensor_names'])).reshape(data_shape)
        gen_module.set_input(key=0, value=data)
        start = time.time()
        gen_module.run()
        end = time.time()
        processing_time += end - start
        output = gen_module.get_output(0).numpy()
        predictions.append(np.argmax(output[0]))
    
    # --- calculate the average processing time ---
    average_processing_time = processing_time / len(df_input)
    print(f'Average processing time: {average_processing_time}')
    
    precision, recall, f1, _ = precision_recall_fscore_support(df_input['labels'], predictions, average='macro')
    accuracy = accuracy_score(df_input['labels'], predictions)
    
    # --- Save Results ---
    result = [{
        'model': 'TVM-ACL_SimpleCNN_CIFAR10',
        'dataset': 'cifar10',
        'task': 'image_classification',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'framework': 'TVM (Arm Compute Library)',
        'framerate': 1.0 / average_processing_time,
    }]
    pd.DataFrame(result).to_csv('result.csv', index=False)
    print(pd.DataFrame(result))

if __name__ == '__main__':
    main()
