## YOLOv8 object tracking with CLI(Command Line Interface)

**Usage**

Pip install the ultralytics package including all requirements.

    pip install ultralytics  

  
  
Run this command in CLI.  

    yolo track  model=yolov8n.pt source='test_video.mp4' show=True   
  

**Models**

| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |  
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |  
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640 | 37.3 | 80.4 | 0.99 | 3.2 | 8.7 |  
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640 | 44.9 | 128.4 | 1.20 | 11.2 | 28.6 |  
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640 | 50.2 | 234.7 | 1.83 | 25.9 | 78.9 |  
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640 | 52.9 | 375.2 | 2.39 | 43.7 | 165.2 |  
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640 | 53.9 | 479.1 | 3.53 | 68.2 | 257.8 |  
  

**Arguments**

- source	'ultralytics/assets'	source directory for images or videos
- conf	0.25	object confidence threshold for detection
- iou	0.7	intersection over union (IoU) threshold for NMS
- half	False	use half precision (FP16)
- device	None	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
- show	False	show results if possible
- save	False	save images with results
- save_txt	False	save results as .txt file
- save_conf	False	save results with confidence scores
- save_crop	False	save cropped images with results
- hide_labels	False	hide labels
- hide_conf	False	hide confidence scores
- max_det	300	maximum number of detections per image
- vid_stride	False	video frame-rate stride
- line_thickness	3	bounding box thickness (pixels)
- visualize	False	visualize model features
- augment	False	apply image augmentation to prediction sources
- agnostic_nms	False	class-agnostic NMS
- retina_masks	False	use high-resolution segmentation masks
- classes	None	filter results by class, i.e. class=0, or class=[0,2,3]
- boxes True Show boxes in segmentation predictions
