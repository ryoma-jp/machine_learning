
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

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking script')

    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    return parser.parse_args()

def load_datasets(dataset_dir):
    dataloader = DataLoader(dataset_name='coco2014_pytorch', dataset_dir=dataset_dir, resize=(320, 320))
    
    return dataloader

def load_model(model):
    if (model == 'ssdlite320_mobilenet_v3_large'):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        
    return model

def predict(device, model, dataloader, output_dir):
    def _decode_predictions(predictions, image_ids, save_file):
        predictions_decoded = []
        for prediction, image_id in zip(predictions, image_ids):
            prediction_decoded = [{'image_id': image_id.detach().cpu().tolist(), 'category_id': category_id.detach().cpu().tolist(), 'bbox': bbox.detach().cpu().tolist(), 'score': score.detach().cpu().numpy()} for category_id, bbox, score in zip(prediction['labels'], prediction['boxes'], prediction['scores'])]
            predictions_decoded += prediction_decoded
        pickle.dump(predictions_decoded, open(save_file, 'wb'))
        
        return
    
    # --- Predict ---
    model.to(device)
    model.eval()
    for i, batch in enumerate(tqdm(dataloader.dataset.testloader)):
        batch_images = torch.Tensor(batch[0]).to(device)
        batch_image_ids = batch[1]
        
        predictions = model(batch_images)
        save_dir = Path(output_dir, 'batch_predictions')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = Path(save_dir, f'predictions_batch{i:06d}.pkl')
        _decode_predictions(predictions, batch_image_ids, save_file)

    return

def evaluate(predictions, annotations):
    pass

def benchmark(args, device):
    # --- Load Arguments ---
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    
    # --- Load Datasets ---
    dataloader = load_datasets(dataset_dir)
    
    # --- Load Model ---
    models = ['ssdlite320_mobilenet_v3_large']
    model = load_model(models[0])
    
    # --- Predict ---
    predict(device, model, dataloader, output_dir)

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
