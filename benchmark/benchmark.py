
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import yaml
from data_loader.data_loader import DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from models.pytorch import simple_cnn
from models.pytorch.ssdlite320_mobilenetv3_large import SSDLite320MobileNetv3Large as PyTorchSSDLite320MobileNetv3Large
from models.onnx.ssdlite320_mobilenetv3_large import SSDLite320MobileNetv3Large as ONNXSSDLite320MobileNetv3Large
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking script')

    parser.add_argument('--config', type=str, default='config/config_sample.yaml',
                        help='Configuration file')
    
    return parser.parse_args()

def load_dataset(dataset_name, dataset_dir, transform, batch_size):
    if (dataset_name == 'cifar10'):
        dataloader = DataLoader(dataset_name='cifar10_pytorch', dataset_dir=dataset_dir, transform=transform)
    elif (dataset_name == 'coco2014'):
        dataloader = DataLoader(dataset_name='coco2014_pytorch', dataset_dir=dataset_dir, resize=(320, 320), transform=transform, batch_size=batch_size)
    elif (dataset_name == 'coco2017'):
        dataloader = DataLoader(dataset_name='coco2017_pytorch', dataset_dir=dataset_dir, resize=(320, 320), transform=transform, batch_size=batch_size)
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')
    
    return dataloader

def load_model(model_name, device, framework, model_path, output_dir):
    if (model_name == 'simple_cnn'):
        model_input_size = [1, 3, 32, 32]
        num_classes = 10
        model = simple_cnn.SimpleCNN(device, input_size=model_input_size, num_classes=num_classes, pth_path=model_path)
    elif (model_name == 'ssdlite320_mobilenet_v3_large'):
        model_input_size = [1, 3, 320, 320]
        if (framework == 'PyTorch'):
            model = PyTorchSSDLite320MobileNetv3Large(device, model_input_size, output_dir=output_dir)
        elif (framework == 'ONNX'):
            model = ONNXSSDLite320MobileNetv3Large(device, model_input_size, output_dir=output_dir)
        else:
            raise ValueError(f'Invalid framework: {framework}')
        
    return model, model_input_size

def predict(device, model, dataloader, output_dir):
    # --- Predict ---
    predictions, targets, preprocessing_time = model.predict(dataloader.dataset.testloader, save_dir=output_dir)

    return predictions, targets, preprocessing_time

def coco_evaluate(predictions, dataset_dir, dataset_name):
    # --- Load COCO Annotations ---
    if (dataset_name == 'coco2014'):
        ann_file = f'{dataset_dir}/annotations/instances_val2014.json'
    elif (dataset_name == 'coco2017'):
        ann_file = f'{dataset_dir}/annotations/instances_val2017.json'
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')
    coco_ann = COCO(ann_file)
    
    # --- Convert Predictions to COCO Format ---
    coco_predictions = []
    for prediction in predictions:
        image_id = prediction['image_id']
        for bbox, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            coco_predictions.append({
                'image_id': image_id,
                'category_id': label,
                'bbox': bbox,
                'score': score,
            })
    coco_predictions = coco_ann.loadRes(coco_predictions)
            
    # --- Evaluate ---
    cocoEval = COCOeval(coco_ann, coco_predictions, 'bbox')
    cocoEval.params.imgIds = [prediction['image_id'] for prediction in predictions]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return cocoEval

def benchmark(args, device):
    # --- Load YAML configuration file ---
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # --- Run Benchmark ---
    benchmark_results = []
    for benchmark in config:
        print(benchmark)
        
        benchmark_name = config[benchmark]['name']
        framework = config[benchmark]['framework']
        dataset_name = config[benchmark]['dataset']
        dataset_dir = config[benchmark]['dataset_dir']
        output_dir = config[benchmark]['output_dir']
        task = config[benchmark]['task']
        model_name = config[benchmark]['model_name']
        model_path = config[benchmark]['model_path']
        
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- Load Model ---
        model, model_input_size = load_model(model_name, device, framework, model_path, output_dir)
        
        # --- Load Datasets ---
        batch_size = 32
        if (framework == 'ONNX'):
            batch_size = 1
        dataloader = load_dataset(dataset_name, dataset_dir, model.transform, batch_size)
        
        # --- Predict ---
        predictions, targets, processing_time = predict(device, model, dataloader, output_dir)
        
        # --- Save Results ---
        if (task == 'image_classification'):
            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='macro')
            accuracy = accuracy_score(targets, predictions)
            
            # --- Save Results ---
            benchmark_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'task': task,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'framework': framework,
                'framerate': 1.0 / processing_time['inference'],
            })
        else: # task == 'object_detection'
            for prediction, target in zip(predictions, targets):
                prediction['image_id'] = target['image_id']

                coef_width = target['image_size'][0] / model_input_size[3]
                coef_height = target['image_size'][1] / model_input_size[2]
                boxes = np.array([prediction['boxes'][:, 0]*coef_width, prediction['boxes'][:, 1]*coef_height, prediction['boxes'][:, 2]*coef_width, prediction['boxes'][:, 3]*coef_height]).T
                prediction['boxes'] = boxes

            # --- Evaluate ---
            cocoEval = coco_evaluate(predictions, dataset_dir, dataset_name)

            # --- Save Results ---
            benchmark_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'task': task,
                'framework': framework,
                'AP50': cocoEval.stats[1],
                'AP75': cocoEval.stats[2],
                'framerate': 1.0 / processing_time['inference'],
            })
    
    pd.DataFrame(benchmark_results).to_csv('benchmark_results.csv', index=False)
    print(pd.DataFrame(benchmark_results))
    
def main():
    # --- Parse arguments ---
    args = parse_args()
    
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Benchmark ---
    benchmark(args, device)

if __name__ == '__main__':
    main()
