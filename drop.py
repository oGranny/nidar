import os
import time
import math
from typing import List, Tuple, Dict, Optional
import sys

from flask import json

from src.helper.mavlink import arm, connect_vehicle, get_attitude, get_current_position, get_heading, gps_prearm_check, guided_goto, haversine_m, rtl, set_mode, takeoff, wait_until_reached
from src.helper.servo import drop_packet, reset_servos
from src.scanning.get_coordinates import pixel_to_ground_gps
import numpy as np
import cv2
import onnxruntime as ort
from pymavlink import mavutil

from src.helper.yolo import YoloDetector

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# ==================== USER INPUT ====================

# Primary route points to visit in order (lat, lon, altRelMeters)
WAYPOINTS: List[Tuple[float, float, float]] = [
    (22.255122, 84.907263, 15),
    (22.255063, 84.906966, 15),
    (22.254918, 84.907342, 15),
    (22.254766, 84.906967, 15),
    (22.254668, 84.907258, 15),
]

# Camera source options for Arducam:
#   "libcamera" - Use picamera2 with libcamera (recommended, no v4l2 needed)
#   0 or "/dev/video0" - Use V4L2 interface (requires legacy camera enabled in raspi-config)
#   Check available devices with: ls /dev/video*
CAMERA_SOURCE = "libcamera"  # Change to 0 if you enable legacy camera support

# YOLO model
MODEL_PATH = "models/yolo11s.onnx"

# Drop mechanism configuration
DROP_SERVOS = [6, 10, 7, 9, 8]  # Servo outputs for drop in order

# =================== MAIN FLOW =======================
def main():
    master = connect_vehicle(config)
    gps_prearm_check(master, config)
    set_mode(master, config, "LOITER")
    # arm(master, config)
    time.sleep(config.get("DLY_AFTER_ARM", 2))
    TAKEOFF_ALT = config.get("TAKEOFF_ALT")

    set_mode(master, config, "GUIDED")
    # takeoff(master, config, TAKEOFF_ALT)

    # Prepare camera + detector
    reset_servos(master, config)    

    if CAMERA_SOURCE == "libcamera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.start()
            cap = None
            print("✓ Arducam initialized")
        except ImportError as e:
            print(f"ERROR: picamera2 failed: {e}")
            print("Make sure to run: sudo apt-get install python3-picamera2")
            return
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print("ERROR: Cannot open camera source:", CAMERA_SOURCE)
            return
        picam2 = None
    
    detector = YoloDetector(MODEL_PATH)

    # Track which drop servo to use next
    drop_index = 0
    DETECTION_ATTEMPTS = config.get("DETECTION_ATTEMPTS")
    RTL_AT_END = config.get("RTL_AT_END")

    try:
        for idx, (lat, lon, alt) in enumerate(WAYPOINTS):
            print(f"\n=== Leg {idx+1}/{len(WAYPOINTS)} → ({lat:.7f},{lon:.7f},{alt:.1f}m) ===")
            guided_goto(master, config, lat, lon, alt)
            reached = wait_until_reached(master, config, lat, lon)

            # Run YOLO multiple times at this waypoint
            print(f"Running YOLO detection at point (up to {DETECTION_ATTEMPTS} attempts)…")
            person_detected = False
            
            for attempt in range(1, DETECTION_ATTEMPTS + 1):
                print(f"  Attempt {attempt}/{DETECTION_ATTEMPTS}")
                
                # Capture frame
                if picam2 is not None:
                    frame = picam2.capture_array()
                    # Convert RGBA to BGR for OpenCV/YOLO
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("  WARN: No frame from camera; skipping this attempt")
                        time.sleep(config.get("SLP_BTW_ATTEMPTS", 0.5))
                        continue
                
                # Run detection
                dets = detector.infer(frame, config, display=True)
                print(f"  Detections: {dets}")
                
                # Check if person detected
                person_dets = [d for d in dets if d["label"] == "person"]
                if person_dets:
                    # Pick highest confidence person
                    person_dets.sort(key=lambda d: d["conf"], reverse=True)
                    person = person_dets[0]
                    
                    # Calculate center pixel of bounding box
                    x1, y1, x2, y2 = person["xyxy"]
                    pixel_x = (x1 + x2) / 2
                    pixel_y = (y1 + y2) / 2
                    
                    # Get drone position and IMU attitude
                    drone_pos = get_current_position(master, config)
                    heading = get_heading(master, config)
                    attitude = get_attitude(master, config)
                    
                    if drone_pos and heading is not None and attitude is not None:
                        drone_lat, drone_lon, drone_alt = drone_pos
                        drone_roll, drone_pitch, drone_yaw = attitude
                        
                        # Apply camera mount offsets to drone attitude
                        camera_pitch = drone_pitch + config.get("CAMERA_PITCH_OFFSET_DEG")
                        camera_roll = drone_roll + config.get("CAMERA_ROLL_OFFSET_DEG")
                        camera_yaw = drone_yaw + config.get("CAMERA_YAW_OFFSET_DEG")
                        
                        # Debug: print input values
                        print(f"    DEBUG: drone_pos={drone_lat:.7f}, {drone_lon:.7f}, alt={drone_alt:.1f}m")
                        print(f"    DEBUG: drone attitude: roll={drone_roll:.1f}°, pitch={drone_pitch:.1f}°, yaw={drone_yaw:.1f}°")
                        print(f"    DEBUG: camera orientation: pitch={camera_pitch:.1f}°, pixel=({pixel_x:.0f}, {pixel_y:.0f})")
                        
                        # Convert pixel to GPS coordinates using actual drone attitude
                        target_gps = pixel_to_ground_gps(
                            pixel=[pixel_x, pixel_y],
                            config=config,
                            drone_lat=drone_lat,
                            drone_lon=drone_lon,
                            altitude_m=drone_alt,
                            heading_deg=heading,
                            pitch_deg=camera_pitch,
                            roll_deg=camera_roll,
                            yaw_deg=camera_yaw
                        )
                        
                        if target_gps:
                            target_lat, target_lon = target_gps
                            print(f"  ✓ PERSON DETECTED at pixel ({pixel_x:.0f}, {pixel_y:.0f})")
                            print(f"  → GPS: {target_lat:.7f}, {target_lon:.7f}")
                            guided_goto(master, target_lat, target_lon, drone_alt)
                            wait_until_reached(master, target_lat, target_lon)
                            
                            # Drop packet at detected location
                            if drop_index < len(DROP_SERVOS):
                                drop_packet(master, config, DROP_SERVOS[drop_index])
                                drop_index += 1
                            else:
                                print("  WARN: No more drop servos available")
                            
                            person_detected = True
                            break  # Skip remaining attempts
                        else:
                            print("  WARN: Could not calculate ground coordinates (ray doesn't hit ground)")
                    else:
                        print("  WARN: Could not get drone position/heading")
                
                # Small delay between attempts
                if attempt < DETECTION_ATTEMPTS:
                    time.sleep(time.sleep("DLY_BTW_DETECTION_ATTEMPTS"))
            
            if not person_detected:
                print("  No person detected in any attempt; continuing to next waypoint.")

        if RTL_AT_END:
            rtl(master)

        print("\nMission complete.")
    finally:
        time.sleep(config.get("SLP_AFT_MSN_COMP"))
        reset_servos(master)
        if picam2 is not None:
            picam2.stop()
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()