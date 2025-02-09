"""
Sample code to inference image files
"""
import os
import sys
import argparse
import numpy as np
import cv2
import tkinter as tk
import time
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, ImageTk, ImageDraw, ImageFont
from picamera2 import Picamera2
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            VDevice)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.utils import time_function
from inference.perform_inference_yolo_det import perform_inference_yolo_det
from inference.perform_inference_yolov8_seg import perform_inference_yolov8_seg
from inference.perform_inference_deeplab_v3 import perform_inference_deeplab_v3
from data_loader.data_loader import DataLoader

@time_function
def load_model(model_path):
    # Load the Hailo8L model
    hef = HEF(model_path)
    return hef

# Dictionary to map model names to their respective inference functions
MODEL_INFERENCE_FUNCTIONS = {
    "yolov8n.hef": perform_inference_yolo_det,
    "yolox_l_leaky.hef": perform_inference_yolo_det,
    "yolox_s_leaky.hef": perform_inference_yolo_det,
    "yolox_tiny.hef": perform_inference_yolo_det,
    "yolox_nano.hef": perform_inference_yolo_det,
    "yolov8s_seg.hef": perform_inference_yolov8_seg,
    "deeplab_v3_mobilenet_v2.hef": perform_inference_deeplab_v3,
}

@time_function
def perform_inference(model_name, frame, input_shape, infer_pipeline, network_group, input_vstream_info):
    # Select the appropriate inference function based on the model name
    inference_function = MODEL_INFERENCE_FUNCTIONS.get(model_name, perform_inference_yolo_det)
    
    # Perform inference using the selected function
    return inference_function(frame, input_shape, infer_pipeline, network_group, input_vstream_info)

def collate_fn(batch):
    return tuple(zip(*batch))

class PaddingImageTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        img_w, img_h = image.size
        model_input_w, model_input_h = self.size
        scale = min(model_input_w / img_w, model_input_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        image = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', self.size, (114, 114, 114))
        new_image.paste(image, (0, 0))
        return new_image

def main():
    parser = argparse.ArgumentParser(description="Image files inference")
    parser.add_argument('--hef', type=str, required=True, help='Path to the Hailo8L model file')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory to save output images')
    args = parser.parse_args()
    
    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    hef = load_model(args.hef)  # Load the model with the provided path
    hef_name = args.hef.split("/")[-1]
    height, width, _ = hef.get_input_vstream_infos()[0].shape
    input_shape = (height, width)
    print(f"Input shape: {height}x{width}")
    
    # Data Loader
    batch_size = 1
    transform = transforms.Compose([
        PaddingImageTransform((height, width)),
        transforms.ToTensor()
    ])
    dataloader = DataLoader(dataset_name='coco2017_pytorch', dataset_dir=args.image_dir, resize=(height, width), transform=transform, batch_size=batch_size)

    # Configure network groups
    start_time = time.time()
    params = VDevice.create_params()
    target = VDevice(params)
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()
    print(f"Network configuration executed in {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    if (hef_name == "yolov8s_seg.hef"):
        input_quantized = False
        output_quantized = True
    else:
        input_quantized = False
        output_quantized = False
    input_vstreams_params = InputVStreamParams.make(network_group, quantized=input_quantized,
                                                    format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, quantized=output_quantized,
                                                    format_type=FormatType.FLOAT32)
    
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    print(f"Stream parameters setup executed in {time.time() - start_time:.4f} seconds")
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            test_dir = os.path.join(args.image_dir, 'val2017')
            image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in tqdm(image_files):
                frame = cv2.imread(image_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = perform_inference(hef_name, frame, input_shape, infer_pipeline, network_group, input_vstream_info)  # Perform inference
                
                # Save the image with inference results
                output_path = os.path.join(args.output_dir, os.path.basename(image_file))
                frame.save(output_path)

if __name__ == "__main__":
    main()
