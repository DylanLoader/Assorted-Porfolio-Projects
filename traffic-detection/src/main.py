import argparse
from typing import List, Tuple, Dict, Set
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd

# Set Detection parameters
COLORS = sv.ColorPalette.default()
right_lane_poly=[
np.array([
[2206, 1215],[2910, 1239],[2226, 839],[1926, 839],[2202, 1223]
])
]

left_lane_poly=[
np.array([
[914, 1371],[1698, 1371],[1718, 2147],[206, 2151],[910, 1371]
])
]

ZONE_LEFT_POLYGONS = left_lane_poly
ZONE_RIGHT_POLYGONS = right_lane_poly

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {} #{"Lane_id": {"class_id": set(tracker_ids)}}
    def update(self,
               detections_all:sv.Detections,
               detections_zones_left:List[sv.Detections],
               detections_zones_right:List[sv.Detections]
               )-> sv.Detections:
        
        for zone_left_id, detections_zone_left in enumerate(detections_zones_left):
            for _, _, _, class_id, tracker_id in detections_zone_left:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_left_id)
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_left_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_left_id, {}) # Dict operation for outer dict
                    self.counts[zone_left_id].setdefault(class_id, set()) # dict operation for inner dict
                    self.counts[zone_left_id][class_id].add(tracker_id) # set operation of subdict

        for zone_right_id, detections_zone_right in enumerate(detections_zones_right):
            for _, _, _, class_id, tracker_id in detections_zone_right:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_right_id)
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_right_id=1
                    self.counts.setdefault(zone_right_id, {}) # Dict operation for outer dict
                    self.counts[zone_right_id].setdefault(class_id, set()) # dict operation for inner dict
                    self.counts[zone_right_id][class_id].add(tracker_id) # set operation of subdict

        return detections_all[detections_all.class_id!=-1]

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position,
)->List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=frame_resolution_wh,
        triggering_position=triggering_position,
        ) for polygon in polygons
    ]

class VideoProcessor: 
    def __init__(
        self,
        source_weights_path:str,
        source_video_path:str,
        target_video_path:str = None,
        confidence_threshold:float=0.3,
        iou_threshold:float=0.7,
    )->None:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.source_weights_path = source_weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(source_weights_path)
        self.box_annotator = sv.BoxAnnotator(
            color=COLORS,
            thickness=4,
            text_thickness=4,
            text_scale=2,
            )
        self.tracker = sv.ByteTrack()
        self.detections_manager = DetectionsManager()
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        self.zones_left = initiate_polygon_zones(
            polygons=ZONE_LEFT_POLYGONS,
            frame_resolution_wh=self.video_info.resolution_wh,
            triggering_position=sv.Position.BOTTOM_CENTER,
        )
        self.zones_right = initiate_polygon_zones(
            polygons=ZONE_RIGHT_POLYGONS,
            frame_resolution_wh=self.video_info.resolution_wh,
            triggering_position=sv.Position.BOTTOM_CENTER,
        )

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info, codec="avc1") as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections)->np.ndarray:
        annotated_frame = frame.copy() # Create local copy of frame to not change in place
        
        for i, zone_left in enumerate(self.zones_left):
            annotated_frame=sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_left.polygon,
                color=COLORS.colors[i],
            )

        for i, zone_right in enumerate(self.zones_right):
            annotated_frame=sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_right.polygon,
                color=COLORS.colors[i+1],
            )

        labels = [
            f"Class: {self.model.model.names[class_id]}, Tracker ID: {tracker_id}"
            for _, _, _, class_id, tracker_id
            in detections
        ]
        annotate_frame= self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
        
        return annotate_frame

    def process_frame(self, frame: np.ndarray)->np.ndarray:
        result = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            )[0]
        detections = sv.Detections.from_ultralytics(result)
        # Filter non-vehicles
        vehicle_classes = [1,2,3,5,6,7]
        valid_detections = np.in1d(detections.class_id, vehicle_classes)
        detections = detections[valid_detections]
        detections = self.tracker.update_with_detections(detections)
        # Set lists to hold detections in zone
        detections_zones_left = []
        detections_zones_right = []

        for zone_left in self.zones_left:
            detections_zone_left = detections[zone_left.trigger(detections=detections)]
            detections_zones_left.append(detections_zone_left)
        
        for zone_right in self.zones_right:
            detections_zone_right = detections[zone_right.trigger(detections=detections)]
            detections_zones_right.append(detections_zone_right)

        # detections = sv.Detections.merge(detections_zones_left)
        detections = self.detections_manager.update(
            detections_all=detections,
            detections_zones_left=detections_zones_left,
            detections_zones_right=detections_zones_right
            )

        return self.annotate_frame(frame=frame, detections=detections)
    
    def get_counts(self)->None:
        current_counts = self.detections_manager.counts
        # Create a data from of the current counts
        left_detections = current_counts[0]
        right_detections = current_counts[1]
        left_dict = {self.model.model.names[k]: len(v) for k, v in left_detections.items()}
        right_dict = {self.model.model.names[k]: len(v) for k, v in right_detections.items()}
        df = pd.DataFrame([left_dict, right_dict], index=['Left', 'Right'])
        # Get the model name 
        source_weights_path = self.source_weights_path.split("/")[-1]
        out_path = f"../data/output/{source_weights_path}.json"
        df.to_json(out_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle counting"
    )
    
    parser.add_argument(
        "--source_weights_path",
        required=True,  
        type=str,
    )
    
    parser.add_argument(
        "--source_video_path",
        required=True,
        type=str,)
    
    parser.add_argument(
        "--target_video_path",
        required=False,
        type=str,)
    
    parser.add_argument(
        "--confidence_threshold", 
        required=False,
        type=float,
    )
    
    parser.add_argument(
        "--iou_threshold",
        type=float,
    )
    
    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
    processor.get_counts()
