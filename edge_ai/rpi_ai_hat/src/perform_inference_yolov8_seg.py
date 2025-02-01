import cv2
import numpy as np
import time
from PIL import Image
from utils import time_function

# Dictionary to store fixed colors for each class
CLASS_COLORS = {}

@time_function
def crop_mask(masks, boxes):
    """
    Zeroing out mask region outside of the predicted bbox.
    https://github.com/hailo-ai/hailo_model_zoo/blob/88365ee4f0d5d2b7a071dcc3bde8db4a81a80009/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py#L446

    Args:
        masks: numpy array of masks with shape [n, h, w]
        boxes: numpy array of bbox coords with shape [n, 4]
    """

    n_masks, _, _ = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)
    for k in range(n_masks):
        masks[k, : y1[k, 0], :] = 0
        masks[k, y2[k, 0] :, :] = 0
        masks[k, :, : x1[k, 0]] = 0
        masks[k, :, x2[k, 0] :] = 0
    return masks

@time_function
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

@time_function
def process_mask(protos, masks_in, bboxes, shape, upsample=True, downsample=False):
    """
    https://github.com/hailo-ai/hailo_model_zoo/blob/88365ee4f0d5d2b7a071dcc3bde8db4a81a80009/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py#L465
    """
    mh, mw, c = protos.shape
    ih, iw = shape
    masks = _sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape((-1, mh, mw))

    downsampled_bboxes = bboxes.copy()
    if downsample:
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        if not masks.shape[0]:
            return None
        masks = cv2.resize(np.transpose(masks, axes=(1, 2, 0)), shape, interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = np.transpose(masks, axes=(2, 0, 1))  # CHW

    if not downsample:
        masks = crop_mask(masks, downsampled_bboxes)  # CHW

    return masks

@time_function
def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    (Created by GitHub Copilot)
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

@time_function
def nms(preds, iou_thres):
    """Perform Non-Maximum Suppression (NMS) on the prediction results.
    (Created by GitHub Copilot)
    
    Args:
        preds (numpy.ndarray): Array of predictions with shape (num_predictions, 5).
                               Each prediction is [x1, y1, x2, y2, confidence].
        iou_thres (float): IoU threshold for NMS.
    
    Returns:
        numpy.ndarray: Indices of the predictions to keep.
    """
    if len(preds) == 0:
        return np.array([])

    # Sort predictions by confidence score in descending order
    indices = np.argsort(preds[:, 4])[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        remaining = indices[1:]
        ious = np.array([iou(preds[current][:4], preds[i][:4]) for i in remaining])

        # Keep only boxes with IoU less than the threshold
        indices = remaining[ious < iou_thres]

    return np.array(keep)

@time_function
def xywh2xyxy(x):
    """
    https://github.com/hailo-ai/hailo_model_zoo/blob/88365ee4f0d5d2b7a071dcc3bde8db4a81a80009/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py#L437
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

@time_function
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=32, multi_label=True):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    https://github.com/hailo-ai/hailo_model_zoo/blob/88365ee4f0d5d2b7a071dcc3bde8db4a81a80009/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py#L351

    Args:
        prediction: numpy.ndarray with shape (batch_size, num_proposals, 351)
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximal number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals
    Returns:
         A list of per image detections, where each is a dictionary with the following structure:
         {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
         }
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, 32)),
                    "detection_classes": np.zeros((0, 80)),
                    "detection_scores": np.zeros((0, 80)),
                }
            )
            continue

        # Confidence = Objectness X Class Score
        x[:, 5:] *= x[:, 4:5]

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label &= nc > 1
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        # per-class NMS
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

        keep = nms(preds, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        if (len(keep) > 0):
            out = x[keep]
            scores = out[:, 4]
            classes = out[:, 5]
            boxes = out[:, :4]
            masks = out[:, 6:]
        else:
            scores = np.zeros((1,))
            classes = np.zeros((1,))
            boxes = np.zeros((1, 4))
            masks = np.zeros((1, 32))

        out = {"detection_boxes": boxes, "mask": masks, "detection_classes": classes, "detection_scores": scores}

        output.append(out)

    return output

@time_function
def letterbox_image(image, size):
    '''
    resize image with unchanged aspect ratio using padding
    https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/dd6ada9d0d10e8b75660b74ab56ba018165204c0/runtime/python/instance_segmentation/yoloseg_inference.py#L98
    '''
    img_w, img_h = image.size
    model_input_w, model_input_h = size
    scale = min(model_input_w / img_w, model_input_h / img_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    image = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (114,114,114))
    new_image.paste(image, ((model_input_w - scaled_w) // 2, (model_input_h - scaled_h) // 2))
    return new_image

@time_function
def perform_inference_yolov8_seg(frame, input_shape, infer_pipeline, network_group, input_vstream_info):
    # Perform inference on the frame using Hailo8L for instance segmentation with YOLOv8
    
    def _softmax(x):
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)
    
    def _yolov8_decoding(raw_boxes, strides, image_dims, reg_max):
        """
        https://github.com/hailo-ai/hailo_model_zoo/blob/88365ee4f0d5d2b7a071dcc3bde8db4a81a80009/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py#L942
        """
        boxes = None
        for box_distribute, stride in zip(raw_boxes, strides):
            # create grid
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # box distribution to distance
            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(
                box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)
            )
            box_distance = _softmax(box_distribute)
            box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1)
            box_distance = box_distance * stride

            # decode box
            box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
            decode_box = np.expand_dims(center, axis=0) + box_distance

            xmin = decode_box[:, :, 0]
            ymin = decode_box[:, :, 1]
            xmax = decode_box[:, :, 2]
            ymax = decode_box[:, :, 3]
            decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])

            xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
            boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
        return boxes  # tf.expand_dims(boxes, axis=2)
    
    # Preprocess the frame
    start_time = time.time()
    height, width = input_shape
    frame_height, frame_width = frame.shape[:2]
    input_tensor = letterbox_image(Image.fromarray(frame), (height, width))
    input_tensor = np.array([input_tensor]).astype(np.float32)
    print(f"Preprocessing executed in {time.time() - start_time:.4f} seconds")
    
    # Perform inference
    start_time = time.time()
    input_tensor = {input_vstream_info.name: np.array([input_tensor]).astype(np.float32)}
    infer_results = infer_pipeline.infer(input_tensor)
    print(f"Inference executed in {time.time() - start_time:.4f} seconds")
    
    # Post-process the output
    #   - Reference
    #     - https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/dd6ada9d0d10e8b75660b74ab56ba018165204c0/runtime/python/instance_segmentation/yoloseg_inference.py
    #     - https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/core/postprocessing/instance_segmentation_postprocessing.py
    start_time = time.time()
    input_names = list(infer_results.keys())

    layer_from_shape: dict = {infer_results[key].shape: key for key in infer_results.keys()}
    mask_channels = 32
    detection_output_channels = 64
    num_classes = 80

    endnodes = [infer_results[layer_from_shape[1, 20, 20, detection_output_channels]],
                infer_results[layer_from_shape[1, 20, 20, num_classes]],
                infer_results[layer_from_shape[1, 20, 20, mask_channels]],
                infer_results[layer_from_shape[1, 40, 40, detection_output_channels]],
                infer_results[layer_from_shape[1, 40, 40, num_classes]],
                infer_results[layer_from_shape[1, 40, 40, mask_channels]],
                infer_results[layer_from_shape[1, 80, 80, detection_output_channels]],
                infer_results[layer_from_shape[1, 80, 80, num_classes]],
                infer_results[layer_from_shape[1, 80, 80, mask_channels]],
                infer_results[layer_from_shape[1, 160, 160, mask_channels]]]
    
    strides = [32,16,8]
    image_dims = (height, width)
    reg_max = 15
    raw_boxes = endnodes[:7:3]
    scores = [np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes)) for s in endnodes[1:8:3]]
    scores = np.concatenate(scores, axis=1)
    outputs = []
    decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, reg_max)
    score_thres = 0.5
    iou_thres = 0.7
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape
    
    # add objectness=1 for working with yolov5_nms
    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1))
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)

    coeffs = [np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]]
    coeffs = np.concatenate(coeffs, axis=1)

    # re-arrange predictions for yolov5_nms
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)

    nms_res = non_max_suppression(predictions, conf_thres=score_thres, iou_thres=iou_thres, multi_label=True)
    masks = []
    outputs = []
    for b in range(batch_size):
        protos = proto_data[b]
        masks = process_mask(protos, nms_res[b]["mask"], nms_res[b]["detection_boxes"], image_dims, upsample=True)
        output = {}
        output["detection_boxes"] = np.array(nms_res[b]["detection_boxes"]) / np.tile(image_dims, 2)
        if masks is not None:
            output["mask"] = np.transpose(masks, (0, 1, 2))
        else:
            output["mask"] = masks
        output["detection_scores"] = np.array(nms_res[b]["detection_scores"])
        output["detection_classes"] = np.array(nms_res[b]["detection_classes"]).astype(int)
        outputs.append(output)
    
    # Create a mask
    input_frame = letterbox_image(Image.fromarray(frame), (height, width))
    mask = np.zeros_like(np.array(input_frame))
    for detection_mask, detection_class in zip(output["mask"], output["detection_classes"]):
        if detection_class not in CLASS_COLORS:
            CLASS_COLORS[detection_class] = tuple(np.random.randint(80, 255, size=3).tolist())
        mask[detection_mask > 0] = np.array(CLASS_COLORS[detection_class])
    
    frame = Image.blend(input_frame, Image.fromarray(mask), alpha=0.5).resize((frame_width, frame_height))
    print(f"Post-processing executed in {time.time() - start_time:.4f} seconds")

    return frame  # Return the frame with detections drawn
