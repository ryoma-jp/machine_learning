
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
from data_loader.data_loader import DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from models.pytorch.ssdlite320_mobilenetv3_large import SSDLite320MobileNetv3Large
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking script')

    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    return parser.parse_args()

def load_dataset(dataset_name, dataset_dir, transform):
    if (dataset_name == 'coco2014'):
        dataloader = DataLoader(dataset_name='coco2014_pytorch', dataset_dir=dataset_dir, resize=(320, 320), transform=transform)
    elif (dataset_name == 'coco2017'):
        dataloader = DataLoader(dataset_name='coco2017_pytorch', dataset_dir=dataset_dir, resize=(320, 320), transform=transform)
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')
    
    return dataloader

def load_model(model_name, device):
    if (model_name == 'ssdlite320_mobilenet_v3_large'):
        model_input_size = [1, 3, 320, 320]
        model = SSDLite320MobileNetv3Large(device, model_input_size)
        
    return model, model_input_size

def predict(device, model, dataloader, output_dir):
    # --- Predict ---
    model.net.to(device)
    predictions, targets = model.predict(dataloader.dataset.testloader)

    return predictions, targets

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
    # --- Load Arguments ---
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    
    # --- Load Model ---
    models = ['ssdlite320_mobilenet_v3_large']
    model, model_input_size = load_model(models[0], device)
    
    datasets = ['coco2014', 'coco2017']
    benchmark_results = []
    for model_name in models:
        for dataset_name in datasets:
            # --- Load Datasets ---
            dataloader = load_dataset(dataset_name, dataset_dir, model.transform)
            
            # --- Predict ---
            predictions, targets = predict(device, model, dataloader, output_dir)
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
                'task': 'object_detection',
                'framework': 'pytorch',
                'AP50': cocoEval.stats[1],
                'AP75': cocoEval.stats[2],
                'framerate': 'T.B.D',
            })
    
    print(pd.DataFrame(benchmark_results))
    
def main():
    # --- Parse arguments ---
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Benchmark ---
    benchmark(args, device)

if __name__ == '__main__':
    main()
