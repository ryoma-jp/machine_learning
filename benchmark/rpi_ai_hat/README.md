# Raspberry Pi AI HAT+

## Setup

see [Getting Started](https://www.raspberrypi.com/documentation/computers/ai.html)

## Demos

use `/user/share/rpicam-camera-assets/*.json` instead of `~/rpicam-apps/assets/*.json`

### Object Detection

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolov6_inference.json --lores-width 640 --lores-height 640
```

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolov8_inference.json --lores-width 640 --lores-height 640
```

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolox_inference.json --lores-width 640 --lores-height 640
```

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolov5_personface.json --lores-width 640 --lores-height 640
```

### Image Detection

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolov5_segmentation.json --lores-width 640 --lores-height 640 --framerate 20
```

### Pose Estimation

```
rpicam-hello -t 0 --post-process-file /user/share/rpicam-camera-assets/hailo_yolov8_pose.json --lores-width 640 --lores-height 640
```

## [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master)

[How to Set Up Raspberry Pi 5 and Hailo](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/install-raspberry-pi5.md#how-to-set-up-raspberry-pi-5-and-hailo-8l)

## References

* [Tutorial of AI Kit with Raspberry Pi 5 about YOLOv8n object detection](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)
