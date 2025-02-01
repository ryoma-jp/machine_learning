"""
Sample code to stream camera feed using OpenCV
"""
import os
import sys
import argparse
import numpy as np
import cv2
import tkinter as tk
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
from picamera2 import Picamera2
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            VDevice)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import time_function
from perform_inference_yolo_det import perform_inference_yolo_det
from perform_inference_yolov8_seg import perform_inference_yolov8_seg
from perform_inference_deeplab_v3 import perform_inference_deeplab_v3

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

def main():
    parser = argparse.ArgumentParser(description="Camera streaming with inference")
    parser.add_argument('--hef', type=str, required=True, help='Path to the Hailo8L model file')
    args = parser.parse_args()
    
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))
    camera.start()
    
    hef = load_model(args.hef)  # Load the model with the provided path
    hef_name = args.hef.split("/")[-1]
    
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
    
    height, width, _ = hef.get_input_vstream_infos()[0].shape
    input_shape = (height, width)
    print(f"Input shape: {height}x{width}")
    
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    print(f"Stream parameters setup executed in {time.time() - start_time:.4f} seconds")
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            root = tk.Tk()
            root.title("Camera Streaming")
            label = tk.Label(root)
            label.pack()
            
            update_camera(hef_name, camera, label, input_shape, infer_pipeline, network_group, input_vstream_info)
            root.mainloop()

@time_function
def update_camera(model_name, camera, label, input_shape, infer_pipeline, network_group, input_vstream_info):
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    frame = perform_inference(model_name, frame, input_shape, infer_pipeline, network_group, input_vstream_info)  # Perform inference
    
    frame = ImageTk.PhotoImage(frame)
    label.config(image=frame)
    label.image = frame
    label.after(10, lambda: update_camera(model_name, camera, label, input_shape, infer_pipeline, network_group, input_vstream_info))
    
if __name__ == "__main__":
    main()
