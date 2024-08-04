
import argparse
import numpy as np
import torch
import torchvision
from data_loader.data_loader import DataLoader
from PIL import Image

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
    def _decode_predictions(predictions):
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        
        return boxes, labels, scores
    
    batch = next(iter(dataloader.dataset.testloader))
    batch_images = batch[0]
    
    # --- save image for debug ---
#    image = Image.fromarray(dataloader.dataset.inverse_normalize(batch[0][0]).numpy().transpose(1, 2, 0).astype('uint8'))
#    print(np.array(image, dtype=np.float32))
#    image.save('test.jpg')
    
    model.eval()
    predictions = model(batch_images)
    print(len(predictions))
    print(predictions[0].keys())
    
    return predictions

def benchmark():
    dataloader = load_datasets()
    
    models = ['ssdlite320_mobilenet_v3_large']
    
    model = load_model(models[0])
    predictions = predict(model, dataloader)
    #print(predictions)

def main():
    # --- Parse arguments ---
    args = parse_args()
    
    # --- Benchmark ---
    benchmark()

if __name__ == '__main__':
    main()
