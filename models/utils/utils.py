
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_detection(detections, image, id, width, height, min_score=0.45):
    def draw_detection_sub(draw, d, c, s, color, scale_factor):
        """Draw box and label for 1 detection."""
        # (Tentative) same as coco.txt
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
        x, y, w, h = d
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        
        font = ImageFont.truetype("LiberationSans-Regular.ttf", size=12)
        draw.rectangle([(xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)], outline=color, width=2)
        text_bbox = draw.textbbox((xmin * scale_factor + 4, ymin * scale_factor + 4), label, font=font)
        text_bbox = list(text_bbox)
        text_bbox[0] -= 4
        text_bbox[1] -= 4
        text_bbox[2] += 4
        text_bbox[3] += 4
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * scale_factor + 4, ymin * scale_factor + 4), label, fill="black", font=font)
        
        return label
    
    boxes = np.array(detections['boxes'])
    classes = np.array(detections['labels']).astype(int)
    if 'scores' in detections:
        scores = np.array(detections['scores'])
    else:
        # onehot from classes
        class_num = 80
        #scores = [np.eye(class_num).astype(float).tolist()[category_id] for category_id in classes]
        scores = [1.0 for _ in range(len(boxes))]
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    scale_factor = max(im_width / width, im_height / height)
    class_colors = {}
    print(f'scale_factor: {scale_factor}')

    num_detections = len(detections['boxes'])
    for idx in range(num_detections):
        if scores[idx] >= min_score:
            if classes[idx] not in class_colors:
                # Assign a random color if not already assigned
                class_colors[classes[idx]] = tuple(np.random.randint(90, 190, size=3).tolist())
            color = class_colors[classes[idx]]
            #scaled_box = [x*width if i%2 else x*height for i,x in enumerate(boxes[idx])]
            scaled_box = boxes[idx]
            label = draw_detection_sub(draw, scaled_box , classes[idx], scores[idx]*100.0, color, scale_factor)
    return image
