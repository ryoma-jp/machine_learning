import cv2
import numpy as np
import time
from PIL import Image
from utils import time_function

# Dictionary to store fixed colors for each class
CLASS_COLORS = {}

@time_function
def perform_inference_deeplab_v3(frame, input_shape, infer_pipeline, network_group, input_vstream_info):
    # Perform inference on the frame using Hailo8L for semantic segmentation with DeeplabV3
    
    # Preprocess the frame
    start_time = time.time()
    height, width = input_shape
    frame_height, frame_width = frame.shape[:2]
    input_tensor = cv2.resize(frame, (height, width))
    input_tensor = np.array([input_tensor]).astype(np.float32)
    print(f"Preprocessing executed in {time.time() - start_time:.4f} seconds")
    
    # Perform inference
    start_time = time.time()
    input_tensor = {input_vstream_info.name: np.array([input_tensor]).astype(np.float32)}
    infer_results = infer_pipeline.infer(input_tensor)
    print(f"Inference executed in {time.time() - start_time:.4f} seconds")
    
    # Post-process the output
    start_time = time.time()
    input_name = list(infer_results.keys())[0]
    output_tensor = infer_results[input_name][0]
    output_tensor = np.argmax(output_tensor, axis=-1)
    
    # Create a mask
    input_frame = Image.fromarray(frame).resize((height, width))
    mask = np.zeros_like(np.array(input_frame))
    for i in range(21):
        if int(i) not in CLASS_COLORS:
            if int(i) == 0:
                CLASS_COLORS[int(i)] = (0, 0, 0)
            else:
                CLASS_COLORS[int(i)] = tuple(np.random.randint(21, 255, size=3).tolist())
        mask[output_tensor == i] = np.array(CLASS_COLORS[int(i)])
    
    frame = Image.blend(input_frame, Image.fromarray(mask), alpha=0.5).resize((frame_width, frame_height))
    print(f"Post-processing executed in {time.time() - start_time:.4f} seconds")

    return frame  # Return the frame with detections drawn
