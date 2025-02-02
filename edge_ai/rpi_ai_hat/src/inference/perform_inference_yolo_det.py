import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from common.utils import time_function

# Dictionary to store fixed colors for each class
CLASS_COLORS = {}

@time_function
def extract_detections(input, boxes, scores, classes, num_detections, threshold=0.5):   
    for i, detection in enumerate(input):
        if len(detection) == 0:
            continue
        for j in range(len(detection)):
            bbox = np.array(detection)[j][:4]
            score = np.array(detection)[j][4]
            if score < threshold:
                continue
            else:
                boxes.append(bbox)
                scores.append(score)
                classes.append(i)
                num_detections = num_detections + 1
    return {'detection_boxes': [boxes], 
              'detection_classes': [classes], 
             'detection_scores': [scores],
             'num_detections': [num_detections]}

@time_function
def post_nms_infer(raw_detections):
    boxes = []
    scores = []
    classes = []
    num_detections = 0
    
    input_name = list(raw_detections.keys())[0]
    detections = extract_detections(raw_detections[input_name][0], boxes, scores, classes, num_detections)
    
    return detections

@time_function
def post_process(detections, image, id, width, height, min_score=0.45):
    global CLASS_COLORS
    boxes = np.array(detections['detection_boxes'])[0]
    classes = np.array(detections['detection_classes'])[0].astype(int)
    scores = np.array(detections['detection_scores'])[0]
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    scale_factor = {
        "width": im_width / width,
        "height": im_height / height,
    }

    for idx in range(np.array(detections['num_detections'])[0]):
        if scores[idx] >= min_score:
            if classes[idx] not in CLASS_COLORS:
                # Assign a random color if not already assigned
                CLASS_COLORS[classes[idx]] = tuple(np.random.randint(90, 190, size=3).tolist())
            color = CLASS_COLORS[classes[idx]]
            scaled_box = [x*width if i%2 else x*height for i,x in enumerate(boxes[idx])]
            label = draw_detection(draw, scaled_box , classes[idx], scores[idx]*100.0, color, scale_factor)
    return image

@time_function
def draw_detection(draw, d, c, s, color, scale_factor):
    """Draw box and label for 1 detection."""
    # same as coco.txt
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    label = class_names[c] + ": " + "{:.2f}".format(s) + '%'
    ymin, xmin, ymax, xmax = d
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=18)
    draw.rectangle([(xmin * scale_factor["width"], ymin * scale_factor["height"]), (xmax * scale_factor["width"], ymax * scale_factor["height"])], outline=color, width=4)
    text_bbox = draw.textbbox((xmin * scale_factor["width"] + 4, ymin * scale_factor["height"] + 4), label, font=font)
    text_bbox = list(text_bbox)
    text_bbox[0] -= 4
    text_bbox[1] -= 4
    text_bbox[2] += 4
    text_bbox[3] += 4
    draw.rectangle(text_bbox, fill=color)
    draw.text((xmin * scale_factor["width"] + 4, ymin * scale_factor["height"] + 4), label, fill="black", font=font)
    return label

@time_function
def padding_image(image, size):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    img_w, img_h = image.size
    model_input_w, model_input_h = size
    scale = min(model_input_w / img_w, model_input_h / img_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    image = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (114,114,114))
    new_image.paste(image, (0, 0))
    return new_image

@time_function
def perform_inference_yolo_det(frame, input_shape, infer_pipeline, network_group, input_vstream_info):
    # Preprocess the frame
    start_time = time.time()
    height, width = input_shape
    input_tensor = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    input_tensor = padding_image(Image.fromarray(input_tensor), (height, width))
    print(f"Preprocessing executed in {time.time() - start_time:.4f} seconds")
    
    # Perform inference
    start_time = time.time()
    input_tensor = {input_vstream_info.name: np.array([input_tensor]).astype(np.float32)}
    print(f"Input tensor: {np.array(input_tensor)}")
    infer_results = infer_pipeline.infer(input_tensor)
    print(f"Inference executed in {time.time() - start_time:.4f} seconds")
    
    # Post-process the output
    start_time = time.time()
    processed_results = post_nms_infer(infer_results)
    image = padding_image(Image.fromarray(frame), (height, width))
    frame = post_process(processed_results, image, 0, height, width)
    print(f"Post-processing executed in {time.time() - start_time:.4f} seconds")
    
    return frame  # Return the frame with detections drawn
