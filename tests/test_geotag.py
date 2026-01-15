#!/usr/bin/env python3
"""
Drone Person Detection & Geotagging using Hailo Object Detection Pipeline

Connects to drone via MAVLink (no control - use transmitter),
runs ML detection using Hailo inference pipeline, and outputs GPS coordinates of detected persons.

Usage:
    python test_geotag.py

Press 'q' to quit.
"""

import json
import math
import os
import sys
import time
import threading
import queue
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from functools import partial

import cv2
import numpy as np
from pymavlink import mavutil

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from src.scanning.transform import pixel_to_ground_gps
from src.helper.mavlink import get_current_position, get_heading, get_attitude
from src.hailo.common.hailo_inference import HailoInfer
from src.hailo.common.toolbox import get_labels, load_json_file
from src.hailo.object_detection.object_detection_post_process import inference_result_handler

# ==================== CONFIGURATION ====================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config1.json')
MODEL_PATH = "models/yolo11s7.hef"
LABELS_PATH = "src/hailo/common/coco.txt"
DETECTION_CONFIG_PATH = "src/hailo/object_detection/config.json"
CAMERA_SOURCE = "camera"  # or 0 for USB camera
BATCH_SIZE = 1

# ==================== LOAD CONFIG ====================
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


class DroneTelemetry:
    """Read-only MAVLink telemetry reader."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.master = None
        self.connected = False
        
        # Cached telemetry
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.heading = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self._running = False
        self._thread = None
    
    def connect(self) -> bool:
        """Connect to drone (read-only)."""
        try:
            print(f"Connecting to drone at {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string)
            self.master.wait_heartbeat(timeout=10)
            print(f"✓ Connected! SYS={self.master.target_system} COMP={self.master.target_component}")
            self.connected = True
            
            # Start background telemetry thread
            self._running = True
            self._thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self._thread.start()
            
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def _telemetry_loop(self):
        """Background thread to read telemetry."""
        while self._running:
            try:
                # GPS
                msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.1)
                if msg:
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.relative_alt / 1000.0
                
                # Heading
                msg = self.master.recv_match(type='VFR_HUD', blocking=True, timeout=0.1)
                if msg:
                    self.heading = float(msg.heading)
                
                # Attitude
                msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
                if msg:
                    self.roll = math.degrees(msg.roll)
                    self.pitch = math.degrees(msg.pitch)
                    self.yaw = math.degrees(msg.yaw)
                    if self.yaw < 0:
                        self.yaw += 360
            except Exception:
                pass
    
    def get_telemetry(self) -> Tuple[float, float, float, float, float, float, float]:
        """Returns (lat, lon, alt, heading, roll, pitch, yaw)"""
        return (self.lat, self.lon, self.alt, self.heading, self.roll, self.pitch, self.yaw)
    
    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.master:
            self.master.close()


class GeotaggingVisualizer:
    """Handles detection result processing and geotagging visualization."""
    
    def __init__(self, drone_telemetry: DroneTelemetry, config: dict, labels: List[str], detection_config: dict):
        self.drone = drone_telemetry
        self.config = config
        self.labels = labels
        self.detection_config = detection_config
        self.detection_count = 0
    
    def process_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Process detections, perform geotagging, and draw on frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries from inference
        
        Returns:
            Annotated frame
        """
        # Get telemetry
        lat, lon, alt, heading, roll, pitch, yaw = self.drone.get_telemetry()
        
        # Filter for persons (class_id 0 in COCO)
        person_dets = [d for d in detections if d.get("class_id") == 0]
        
        if person_dets:
            self.detection_count += 1
            print(f"\n--- Detection #{self.detection_count} ---")
            print(f"Drone: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.1f}m, hdg={heading:.0f}°")
            print(f"Attitude: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
            print(f"✓ Found {len(person_dets)} person(s)")
            
            for i, person in enumerate(person_dets):
                # Extract bbox coordinates
                bbox = person.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    pixel_x = (x1 + x2) / 2
                    pixel_y = (y1 + y2) / 2
                    conf = person.get("confidence", 0.0)
                    
                    print(f"\n  Person {i+1}: conf={conf:.2f}, pixel=({pixel_x:.0f}, {pixel_y:.0f})")
                    
                    # Apply camera mount offsets
                    camera_pitch = pitch + self.config.get("CAMERA_PITCH_OFFSET_DEG", -90)
                    camera_roll = roll + self.config.get("CAMERA_ROLL_OFFSET_DEG", 0)
                    camera_yaw = yaw + self.config.get("CAMERA_YAW_OFFSET_DEG", 0)
                    
                    # Transform pixel to GPS
                    if alt > 0.0:  # Only if airborne
                        target_gps = pixel_to_ground_gps(
                            pixel=[pixel_x, pixel_y],
                            config=self.config,
                            drone_lat=lat,
                            drone_lon=lon,
                            altitude_m=alt,
                            roll_deg=camera_roll,
                            pitch_deg=camera_pitch,
                            yaw_deg=camera_yaw
                        )
                        
                        if target_gps:
                            target_lat, target_lon = target_gps
                            print(f"  📍 GPS: {target_lat:.7f}, {target_lon:.7f}")
                            
                            # Draw GPS on frame
                            cv2.putText(frame, f"GPS: {target_lat:.6f}, {target_lon:.6f}", 
                                       (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (0, 255, 255), 2)
                        else:
                            print(f"  ⚠ Ray doesn't hit ground")
                    else:
                        print(f"  ⚠ Drone not airborne (alt={alt:.1f}m)")
                    
                    # Draw bbox and center point
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Person {conf:.2f}", 
                               (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
        
        # Draw telemetry overlay
        overlay = [
            f"GPS: {lat:.6f}, {lon:.6f}",
            f"Alt: {alt:.1f}m  Hdg: {heading:.0f}",
            f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}",
            f"Detections: {len(person_dets)}"
        ]
        y = 30
        for line in overlay:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        
        return frame


def preprocess_frame(frame, width, height):
    """Preprocess a single frame for inference."""
    resized = cv2.resize(frame, (width, height))
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb_frame


def inference_callback(completion_info, bindings_list: list, input_batch: list, 
                       output_queue: queue.Queue, labels: List[str], 
                       detection_config: dict) -> None:
    """
    Callback to handle inference results and extract detections.
    """
    if completion_info.exception:
        print(f'Inference error: {completion_info.exception}')
        output_queue.put((input_batch[0], []))
    else:
        for i, bindings in enumerate(bindings_list):
            # Get raw output
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            
            # Process detections using the post-process handler
            # This will return annotated frame and detections
            detections = extract_detections_from_result(result, labels, detection_config)
            
            output_queue.put((input_batch[i], detections))


def extract_detections_from_result(result, labels, detection_config):
    """
    Extract detection information from inference result.
    Returns list of detection dictionaries.
    """
    detections = []
    
    # Parse the result based on the model output format
    # This is a simplified version - adjust based on your model's output
    if isinstance(result, dict):
        # Multi-output model (e.g., YOLOv8)
        # Process output tensors to extract detections
        # This needs to be adapted to your specific model output format
        pass
    else:
        # Single output tensor
        # Parse detections from the tensor
        pass
    
    return detections


def main():
    print("="*60)
    print("PERSON DETECTION & GEOTAGGING (HAILO PIPELINE)")
    print("Control drone with transmitter - this script only observes")
    print("="*60)
    
    # Connect to drone
    drone = DroneTelemetry(config.get("MAVPROXY_URL", "udp:127.0.0.1:14550"))
    if not drone.connect():
        print("⚠ Running without drone telemetry")
    
    # Load labels and config
    labels = get_labels(LABELS_PATH)
    detection_config = load_json_file(DETECTION_CONFIG_PATH)
    
    # Initialize camera
    picam2 = None
    cap = None
    
    if CAMERA_SOURCE == "camera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.start()
            print("✓ Picamera2 initialized")
        except Exception as e:
            print(f"✗ Picamera2 failed: {e}")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE if isinstance(CAMERA_SOURCE, int) else 0)
        if not cap.isOpened():
            print(f"✗ Cannot open camera: {CAMERA_SOURCE}")
            return
        print("✓ Camera initialized")
    
    # Load Hailo model
    print(f"Loading model: {MODEL_PATH}")
    hailo_inference = HailoInfer(MODEL_PATH, BATCH_SIZE)
    height, width, _ = hailo_inference.get_input_shape()
    print(f"✓ Hailo model loaded (input: {width}x{height})")
    
    # Initialize geotagging visualizer
    visualizer = GeotaggingVisualizer(drone, config, labels, detection_config)
    
    print("\n" + "="*60)
    print("RUNNING - Press 'q' to quit")
    print("="*60 + "\n")
    
    # Queues for pipeline
    input_queue = queue.Queue(maxsize=10)
    output_queue = queue.Queue(maxsize=10)
    
    # Flag to control threads
    running = True
    
    def inference_thread():
        """Thread for running inference."""
        while running:
            try:
                next_batch = input_queue.get(timeout=0.1)
                if next_batch is None:
                    break
                
                original_frame, preprocessed_frame = next_batch
                
                # Create callback with bound parameters
                callback_fn = partial(
                    inference_callback,
                    input_batch=[original_frame],
                    output_queue=output_queue,
                    labels=labels,
                    detection_config=detection_config
                )
                
                # Run inference
                hailo_inference.run([preprocessed_frame], callback_fn)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
    
    # Start inference thread
    infer_thread = threading.Thread(target=inference_thread, daemon=True)
    infer_thread.start()
    
    try:
        while True:
            # Capture frame
            if picam2 is not None:
                frame = picam2.capture_array()
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            
            # Preprocess frame
            preprocessed = preprocess_frame(frame, width, height)
            
            # Put in input queue (non-blocking)
            try:
                input_queue.put_nowait((frame.copy(), preprocessed))
            except queue.Full:
                pass  # Skip frame if queue is full
            
            # Try to get result from output queue
            try:
                original_frame, detections = output_queue.get_nowait()
                
                # Process detections and draw geotagging
                annotated_frame = visualizer.process_detections(original_frame, detections)
                
                # Show frame
                cv2.imshow("Person Geotagging (Hailo) - Press 'q' to quit", annotated_frame)
            except queue.Empty:
                # No result yet, show original frame
                cv2.imshow("Person Geotagging (Hailo) - Press 'q' to quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\nShutting down...")
        running = False
        input_queue.put(None)  # Signal inference thread to stop
        infer_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        
        hailo_inference.close()
        drone.close()
        print("✓ Done")


if __name__ == "__main__":
    main()
