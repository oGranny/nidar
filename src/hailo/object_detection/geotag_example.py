#!/usr/bin/env python3
"""
Example: Geotagging tracked persons with MAVLink telemetry

This example shows how to integrate the object detection pipeline
with drone telemetry to geotag detected persons.

Usage:
    python geotag_example.py -n ~/hailo-apps/yolo8s.hef -i camera --track

Requirements:
    - Drone connected via MAVLink (e.g., via MAVProxy)
    - Camera connected
    - Hailo device
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_detection import run_inference_pipeline, parse_args
from object_detection_post_process import (
    set_track_lost_callback,
    set_drone_telemetry_getter
)

# Try to import MAVLink helper
try:
    from src.helper.mavlink import (
        connect_vehicle,
        get_current_position,
        get_heading,
        get_attitude
    )
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("Warning: MAVLink helper not found. Geotagging will be disabled.")

# Try to import geotag persistence
try:
    from src.scanning.geotag import save_coordinate_with_threshold
    GEOTAG_DB_AVAILABLE = True
except ImportError:
    GEOTAG_DB_AVAILABLE = False


# Global MAVLink connection
_master = None
_config = {}


def init_mavlink(config_path: str = None) -> bool:
    """Initialize MAVLink connection to drone."""
    global _master, _config
    
    if not MAVLINK_AVAILABLE:
        return False
    
    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            _config = json.load(f)
    else:
        # Default config
        _config = {
            "MAVPROXY_URL": "udp:127.0.0.1:14550",
            "DLY_POS_CHK_GUIDED": 1
        }
    
    try:
        _master = connect_vehicle(_config)
        print("✓ MAVLink connected!")
        return True
    except Exception as e:
        print(f"✗ MAVLink connection failed: {e}")
        return False


def get_drone_telemetry() -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    Get current drone telemetry.
    
    Returns:
        Tuple of (lat, lon, alt, heading, roll, pitch, yaw) or None
    """
    global _master, _config
    
    if _master is None:
        return None
    
    try:
        # Get position
        position = get_current_position(_master, _config)
        if position is None:
            return None
        lat, lon, alt = position
        
        # Get heading
        heading = get_heading(_master, _config)
        if heading is None:
            heading = 0.0
        
        # Get attitude
        attitude = get_attitude(_master, _config)
        if attitude is None:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        else:
            roll, pitch, yaw = attitude
        
        return (lat, lon, alt, heading, roll, pitch, yaw)
    except Exception as e:
        print(f"Telemetry error: {e}")
        return None


def on_person_lost(track_id: int, track_data: dict) -> None:
    """
    Callback when a tracked person leaves the frame.
    
    This function is called automatically when ByteTrack loses a track.
    You can customize this to:
    - Save the person's location to a database
    - Send an alert
    - Log the event
    - Navigate the drone to that location
    
    Args:
        track_id: The unique track ID that was lost
        track_data: Dictionary containing:
            - 'geotags': list of (lat, lon) positions
            - 'centroids': list of (x, y) pixel positions  
            - 'class': detected class name (e.g., 'person')
            - 'final_gps': averaged GPS position (if available)
    """
    print(f"\n{'='*60}")
    print(f"🎯 PERSON TRACKING COMPLETE")
    print(f"   Track ID: {track_id}")
    print(f"   Class: {track_data.get('class', 'unknown')}")
    print(f"   Total frames tracked: {len(track_data.get('centroids', []))}")
    
    final_gps = track_data.get('final_gps')
    if final_gps:
        lat, lon = final_gps
        print(f"   📍 Final GPS: {lat:.7f}, {lon:.7f}")
        
        # Save to database if available
        if GEOTAG_DB_AVAILABLE:
            try:
                result = save_coordinate_with_threshold(
                    coordinate=[lat, lon],
                    threshold_meters=5.0  # Merge points within 5m
                )
                print(f"   💾 Saved to database: {result}")
            except Exception as e:
                print(f"   ⚠ Database save failed: {e}")
        
        # You can add more actions here:
        # - Send to ground station
        # - Add to search grid
        # - Trigger drone navigation
        # Example: guided_goto(_master, _config, lat, lon, current_alt)
        
    else:
        print(f"   ⚠ No GPS data available (drone telemetry not connected?)")
    
    print(f"{'='*60}\n")


def main():
    """Main entry point with geotagging enabled."""
    
    # Parse standard arguments
    args = parse_args()
    
    # Initialize MAVLink if tracking is enabled
    if args.track:
        print("\n--- Initializing Geotagging System ---")
        
        # Try to connect to drone
        mavlink_ok = init_mavlink()
        
        if mavlink_ok:
            # Set up telemetry getter
            set_drone_telemetry_getter(get_drone_telemetry)
            print("✓ Telemetry getter configured")
        else:
            print("⚠ Running without live GPS - using test coordinates")
            # You can set up mock telemetry for testing:
            # set_drone_telemetry_getter(lambda: (28.6139, 77.2090, 50.0, 0.0, 0.0, 0.0, 0.0))
        
        # Set up lost track callback
        set_track_lost_callback(on_person_lost)
        print("✓ Track lost callback configured")
        print("-----------------------------------\n")
    
    # Run the inference pipeline
    run_inference_pipeline(
        args.net, args.input, args.batch_size, args.labels,
        args.output_dir, args.save_stream_output, args.camera_resolution,
        args.output_resolution, args.track, args.show_fps, args.framerate, 
        args.draw_trail
    )


if __name__ == "__main__":
    main()
