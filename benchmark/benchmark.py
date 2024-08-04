
import argparse
import numpy as np
import pickle
import torch
import torchvision
from data_loader.data_loader import DataLoader
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking script')

    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    return parser.parse_args()

def load_datasets():
    dataset_dir = './dataset'
    dataloader = DataLoader(dataset_name='coco2014_pytorch', dataset_dir=dataset_dir)
    
    return dataloader

def load_model(model):
    if (model == 'ssdlite320_mobilenet_v3_large'):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        
    return model

def predict(model, dataloader):
    def _decode_predictions(predictions, image_ids):
        predictions_decoded = []
        for prediction, image_id in zip(predictions, image_ids):
            prediction_decoded = [{'image_id': image_id.detach().tolist(), 'category_id': category_id.detach().tolist(), 'bbox': bbox.detach().tolist(), 'score': score.detach().numpy()} for category_id, bbox, score in zip(prediction['labels'], prediction['boxes'], prediction['scores'])]
            predictions_decoded += prediction_decoded
        
        return predictions_decoded
    
    model.eval()
    
    predictions_decoded = []
    for batch in tqdm(dataloader.dataset.testloader):
        batch_images = batch[0]
        batch_image_ids = batch[1]
        
        predictions_decoded += _decode_predictions(model(batch_images), batch_image_ids)
    
    return predictions_decoded

def evaluate(predictions, annotations):
    pass

def benchmark():
    dataloader = load_datasets()
    
    models = ['ssdlite320_mobilenet_v3_large']
    
    model = load_model(models[0])
    predictions = predict(model, dataloader)
    print(len(predictions))
    
    pickle.dump(predictions, open('predictions.pkl', 'wb'))

def main():
    # --- Parse arguments ---
    args = parse_args()
    
    # --- Benchmark ---
    benchmark()

if __name__ == '__main__':
    main()
