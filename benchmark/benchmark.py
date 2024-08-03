
import argparse
import torch
import torchvision
from data_loader import datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking script')

    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    return parser.parse_args()

def load_datasets():
    dataset = datasets.Coco2014Dataset(root='./dataset', download=True, train=False)
    
    return dataset

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
