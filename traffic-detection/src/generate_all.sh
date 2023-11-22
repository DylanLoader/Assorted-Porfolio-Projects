python3 main.py \
--source_weights_path ../models/yolov8s.pt \
--source_video_path ../data/vehicle-counting.mp4 \
--confidence_threshold 0.65 \
--iou_threshold 0.5 \
--target_video_path ../data/vehicle-counting-yolov8s.mp4

python3 main.py \
--source_weights_path ../models/yolov8m.pt \
--source_video_path ../data/vehicle-counting.mp4 \
--confidence_threshold 0.65 \
--iou_threshold 0.5 \
--target_video_path ../data/vehicle-counting-yolov8m.mp4

python3 main.py \
--source_weights_path ../models/yolov8l.pt \
--source_video_path ../data/vehicle-counting.mp4 \
--confidence_threshold 0.65 \
--iou_threshold 0.5 \
--target_video_path ../data/vehicle-counting-yolov8l.mp4