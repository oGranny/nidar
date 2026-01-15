#!/usr/bin/env python3
"""
Drone Person Detection & Geotagging (Read-Only Mode)

Connects to drone via MAVLink (no control - use transmitter),
runs ML detection every 2 seconds, and outputs GPS coordinates of detected persons.

Usage:
    python test_transform.py

Press 'q' to quit.
"""

import json
import math
import os
import sys
import time
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from pymavlink import mavutil

# Local imports
from src.helper.yolo import YoloDetector
from src.scanning.transform import pixel_to_ground_gps
from src.helper.mavlink import get_current_position, get_heading, get_attitude

# ==================== CONFIGURATION ====================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
MODEL_PATH = "models/yolo11s.onnx"
CAMERA_SOURCE = "libcamera"  # or 0 for USB camera
DETECTION_INTERVAL = 2.0  # Run detection every 2 seconds

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


def main():
    print("="*60)
    print("PERSON DETECTION & GEOTAGGING (READ-ONLY MODE)")
    print("Control drone with transmitter - this script only observes")
    print("="*60)
    
    # Connect to drone
    drone = DroneTelemetry(config.get("MAVPROXY_URL", "udp:127.0.0.1:14550"))
    if not drone.connect():
        print("⚠ Running without drone telemetry")
    
    # Initialize camera
    picam2 = None
    cap = None
    
    if CAMERA_SOURCE == "libcamera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.start()
            print("✓ Picamera2 initialized")
        except Exception as e:
            print(f"✗ Picamera2 failed: {e}")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print(f"✗ Cannot open camera: {CAMERA_SOURCE}")
            return
        print("✓ Camera initialized")
    
    # Load YOLO detector
    print(f"Loading model: {MODEL_PATH}")
    detector = YoloDetector(MODEL_PATH)
    print("✓ YOLO detector loaded")
    
    print("\n" + "="*60)
    print("RUNNING - Press 'q' to quit")
    print("Detection runs every 2 seconds")
    print("="*60 + "\n")
    
    last_detection_time = 0
    detection_count = 0
    
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
            
            current_time = time.time()
            
            # Get telemetry for display
            lat, lon, alt, heading, roll, pitch, yaw = drone.get_telemetry()
            
            # Run detection every DETECTION_INTERVAL seconds
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                last_detection_time = current_time
                detection_count += 1
                
                print(f"\n--- Detection #{detection_count} ---")
                print(f"Drone: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.1f}m, hdg={heading:.0f}°")
                print(f"Attitude: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
                
                # Run YOLO detection
                dets = detector.infer(frame, config, display=True)
                
                # Filter for persons
                person_dets = [d for d in dets if d["label"] == "person"]
                
                if person_dets:
                    print(f"✓ Found {len(person_dets)} person(s)")
                    
                    for i, person in enumerate(person_dets):
                        x1, y1, x2, y2 = person["xyxy"]
                        pixel_x = (x1 + x2) / 2
                        pixel_y = (y1 + y2) / 2
                        conf = person["conf"]
                        
                        print(f"\n  Person {i+1}: conf={conf:.2f}, pixel=({pixel_x:.0f}, {pixel_y:.0f})")
                        
                        # Apply camera mount offsets
                        camera_pitch = pitch + config.get("CAMERA_PITCH_OFFSET_DEG", -90)
                        camera_roll = roll + config.get("CAMERA_ROLL_OFFSET_DEG", 0)
                        camera_yaw = yaw + config.get("CAMERA_YAW_OFFSET_DEG", 0)
                        
                        # Transform pixel to GPS
                        if alt > 0.0:  # Only if airborne
                            target_gps = pixel_to_ground_gps(
                                pixel=[pixel_x, pixel_y],
                                config=config,
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
                        
                        # Draw bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                else:
                    print("No persons detected")
            
            # Draw telemetry overlay
            overlay = [
                f"GPS: {lat:.6f}, {lon:.6f}",
                f"Alt: {alt:.1f}m  Hdg: {heading:.0f}",
                f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}",
                f"Next detection in: {max(0, DETECTION_INTERVAL - (current_time - last_detection_time)):.1f}s"
            ]
            y = 30
            for line in overlay:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
            
            # Show frame
            cv2.imshow("Person Geotagging - Press 'q' to quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        drone.close()
        print("✓ Done")


if __name__ == "__main__":
    main()
