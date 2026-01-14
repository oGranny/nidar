import cv2
import numpy as np
from common.toolbox import id_to_color

import os
from collections import deque
from typing import Optional, Tuple, Dict, Callable
import sys

# Add parent paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Dictionary to store a limited history of tracklet coordinates.
# The keys will be the track IDs.
tracklet_history = {}
# Maximum number of past frames to display
trail_length = 30 
# Only draw trail for certain classes (e.g., person=0, phone=67 in COCO)
TRACKLET_CLASSES = [0, 67]  # PERSON, SMARTPHONE

# Track active IDs to detect when persons leave frame
_active_track_ids: set = set()
# Store geotagged data for each track
_track_geotags: Dict[int, list] = {}
# Callback for when track is lost
_on_track_lost_callback: Optional[Callable] = None
# Drone telemetry getter function
_get_drone_telemetry: Optional[Callable] = None


def set_track_lost_callback(callback: Callable[[int, dict], None]) -> None:
    """
    Set callback function to run when a tracked person leaves the frame.
    
    Args:
        callback: Function that takes (track_id, track_data) where track_data contains:
                  - 'geotags': list of (lat, lon) positions
                  - 'centroids': list of (x, y) pixel positions
                  - 'class': detected class name
    """
    global _on_track_lost_callback
    _on_track_lost_callback = callback


def set_drone_telemetry_getter(getter: Callable[[], Optional[Tuple[float, float, float, float, float, float, float]]]) -> None:
    """
    Set function to get drone telemetry data.
    
    Args:
        getter: Function that returns (lat, lon, alt, heading, roll, pitch, yaw) or None
    """
    global _get_drone_telemetry
    _get_drone_telemetry = getter


def _geotag_detection(bbox: list, image_shape: tuple) -> Optional[Tuple[float, float]]:
    """
    Convert detection bbox to GPS coordinates using drone telemetry.
    
    Args:
        bbox: [xmin, ymin, xmax, ymax] bounding box
        image_shape: (height, width) of the image
        
    Returns:
        (latitude, longitude) or None if telemetry unavailable
    """
    if _get_drone_telemetry is None:
        return None
    
    telemetry = _get_drone_telemetry()
    if telemetry is None:
        return None
    
    lat, lon, alt, heading, roll, pitch, yaw = telemetry
    
    # Import geo_transform function
    try:
        from geo_transform import person_gps_from_bbox
    except ImportError:
        try:
            from src.geo_transform import person_gps_from_bbox
        except ImportError:
            return None
    
    # Get image dimensions for camera intrinsics
    img_height, img_width = image_shape[:2]
    cx, cy = img_width / 2, img_height / 2
    # Estimate focal length (adjust based on your camera calibration)
    fx = fy = img_width * 0.8  # Approximate for typical drone cameras
    
    xmin, ymin, xmax, ymax = bbox
    
    result = person_gps_from_bbox(
        bbox_x1=xmin, bbox_y1=ymin,
        bbox_x2=xmax, bbox_y2=ymax,
        drone_lat=lat, drone_lon=lon,
        altitude_m=alt, heading_deg=heading,
        roll=roll, pitch=pitch, yaw=yaw,
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    
    return result

def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None, draw_trail=False):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        infer_results (list): Raw output from the model.
        original_frame (np.ndarray): Original image frame.
        labels (list): List of class labels.
        enable_tracking (bool): Whether tracking is enabled.
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    detections = extract_detections(original_frame, infer_results, config_data)  # Should return dict with boxes, classes, scores
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker, draw_trail=draw_trail)
    return frame_with_detections


def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        labels (list): List of labels (1 or 2 elements).
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        track (bool): Whether to include tracking info.
    """
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compose texts
    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = None

    if track:
        if len(labels) == 2:
            bottom_text = labels[1]
        else:
            bottom_text = labels[0]


    # Set colors
    text_color = (255, 255, 255)  # White
    border_color = (0, 0, 0)  # Black

    # Draw top text with black border first
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw bottom text if exists
    if bottom_text:
        pos = (xmax - 50, ymax - 6)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    # Scale box coordinates
    box = [int(x * size) for x in box]

    # Apply padding correction
    for i in range(4):
        if i % 2 == 0:  # x-coordinates
            if input_height != size:
                box[i] -= padding_length
        else:  # y-coordinates
            if input_width != size:
                box[i] -= padding_length

    # Swap to [ymin, xmin, ymax, xmax]
    return [box[1], box[0], box[3], box[2]]


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    # Sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    # Take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None, draw_trail=False) -> np.ndarray:
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """

    # Extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [xmin,ymin,xmaxm, ymax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        # Convert detection format to [xmin,ymin,xmaxm ymax,score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  # [x, y, w, h]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        # Skip tracking if no detections passed
        if not dets_for_tracker:
            # Check for lost tracks when no detections
            _check_lost_tracks(set(), labels, classes if classes else [])
            return img_out

        # Run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))
        
        # Collect current track IDs
        current_track_ids = set()

        # Draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  # Unique tracker ID
            current_track_ids.add(track_id)
            
            x1, y1, x2, y2 = track.tlbr  # Bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            color = tuple(id_to_color(classes[best_idx]).tolist())  # Color based on class
            class_name = labels[classes[best_idx]] if best_idx is not None else "unknown"
            
            if best_idx is None:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
                               track.score * 100.0, color, track=True)
            else:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
                               track.score * 100.0, color, track=True)
            
            # Compute centroid
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centroid = (center_x, center_y)
            
            # Geotag the detection
            geotag = _geotag_detection([xmin, ymin, xmax, ymax], img_out.shape)
            
            # Store geotag data for this track
            if track_id not in _track_geotags:
                _track_geotags[track_id] = {
                    'geotags': [],
                    'centroids': [],
                    'class': class_name
                }
            
            if geotag:
                _track_geotags[track_id]['geotags'].append(geotag)
                print(f"Track ID: {track_id}, Class: {class_name}, Centroid: {centroid}, GPS: ({geotag[0]:.7f}, {geotag[1]:.7f})")
            else:
                print(f"Track ID: {track_id}, Class: {class_name}, Centroid: {centroid}")
            
            _track_geotags[track_id]['centroids'].append(centroid)
                               
            if best_idx is None or classes[best_idx] not in TRACKLET_CLASSES:
                continue

            # Initialize or update the tracklet history
            if track_id not in tracklet_history:
                tracklet_history[track_id] = deque(maxlen=trail_length)
            tracklet_history[track_id].append(centroid)

            if draw_trail:
                for i in range(1, len(tracklet_history[track_id])):
                    # Get the center point for the current and previous frames
                    point_a = tracklet_history[track_id][i-1]
                    point_b = tracklet_history[track_id][i]

                    # Draw a line between the points and draw the points as circles
                    cv2.line(img_out, point_a, point_b, color, 3) #(255, 0, 0), 2)
                    cv2.circle(img_out, point_b, radius=20, thickness=1, color=color) #, thickness=-1) # -1 for filled circle
        
        # Check for lost tracks
        _check_lost_tracks(current_track_ids, labels, classes)



    else:
        # No tracking — draw raw model detections
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())  # Color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)


def _check_lost_tracks(current_ids: set, labels: list, classes: list) -> None:
    """
    Check for tracks that are no longer present and trigger callback.
    
    Args:
        current_ids: Set of currently active track IDs
        labels: List of class labels
        classes: List of class indices
    """
    global _active_track_ids, _track_geotags, _on_track_lost_callback
    
    lost_ids = _active_track_ids - current_ids
    
    for lost_id in lost_ids:
        track_data = _track_geotags.get(lost_id, {
            'geotags': [],
            'centroids': [],
            'class': 'unknown'
        })
        
        # Compute final GPS position (average of all geotags)
        final_gps = None
        if track_data['geotags']:
            avg_lat = sum(g[0] for g in track_data['geotags']) / len(track_data['geotags'])
            avg_lon = sum(g[1] for g in track_data['geotags']) / len(track_data['geotags'])
            final_gps = (avg_lat, avg_lon)
            track_data['final_gps'] = final_gps
        
        print(f"\n{'='*50}")
        print(f"🚨 TRACK LOST: ID {lost_id} ({track_data['class']})")
        if final_gps:
            print(f"📍 Final GPS Position: ({final_gps[0]:.7f}, {final_gps[1]:.7f})")
            print(f"   Total geotag samples: {len(track_data['geotags'])}")
        print(f"{'='*50}\n")
        
        # Call user callback if set
        if _on_track_lost_callback:
            try:
                _on_track_lost_callback(lost_id, track_data)
            except Exception as e:
                print(f"Error in track lost callback: {e}")
        
        # Clean up track data
        if lost_id in _track_geotags:
            del _track_geotags[lost_id]
        if lost_id in tracklet_history:
            del tracklet_history[lost_id]
    
    # Update active track IDs
    _active_track_ids = current_ids.copy()


def get_track_geotag(track_id: int) -> Optional[dict]:
    """
    Get current geotag data for a specific track.
    
    Args:
        track_id: The track ID to query
        
    Returns:
        Dict with 'geotags', 'centroids', 'class', and optionally 'final_gps'
    """
    return _track_geotags.get(track_id)


def get_all_active_tracks() -> Dict[int, dict]:
    """
    Get geotag data for all currently active tracks.
    
    Returns:
        Dict mapping track_id -> track_data
    """
    return _track_geotags.copy()
