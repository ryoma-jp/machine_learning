
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
    
    batch = next(iter(dataloader.dataset.testloader))
    print(batch)
    print(batch[0].shape)
    
    # --- save image for debug ---
    image = Image.fromarray(dataloader.dataset.inverse_normalize(batch[0][0]).numpy().transpose(1, 2, 0).astype('uint8'))
    print(np.array(image, dtype=np.float32))
    image.save('test.jpg')
    
    return dataloader

def load_model(model):
    if (model == 'ssdlite320_mobilenet_v3_large'):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        
    return model

def predict(model, images):
    model.eval()
    x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
    predictions = model(x)
#    predictions = model(images)
    
    return predictions

def benchmark():
    load_datasets()
    
    models = ['ssdlite320_mobilenet_v3_large']
    
#    model = load_model(models[0])
#    predictions = predict(model, None)
#    print(predictions)

def main():
    # --- Parse arguments ---
    args = parse_args()
    
    # --- Benchmark ---
    benchmark()

if __name__ == '__main__':
    main()
