import os
import sys
import threading
import time
from json import load as json_load

from pymavlink import mavutil
# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.navigation.geofence import upload_geofence, kml_to_boundary_coordinates
from src.helper.mavlink import (
    arm, connect_vehicle, gps_prearm_check, set_mode, 
    takeoff, upload_mission, get_current_position, get_attitude
)

# Config paths
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json_load(f)

# Camera config for GPS projection
camera_config_path = os.path.join(os.path.dirname(__file__), 'src', 'hailo', 'object_detection', 'config.json')


# Global flag to stop inference thread
_stop_inference = False


def run_object_detection(master, camera_config, hef_path: str, input_source: str):
    """
    Run object detection inference in a separate thread.
    
    Args:
        master: pymavlink connection for telemetry.
        camera_config: Camera intrinsics for GPS projection.
        hef_path: Path to the HEF model file.
        input_source: Video file or camera index.
    """
    global _stop_inference
    
    # Import here to avoid issues if hailo not available
    try:
        from src.hailo.object_detection.tracklet_database import set_mavlink_master, init_database
        from src.hailo.object_detection.object_detection_post_process import set_camera_config
        from src.hailo.object_detection.object_detection import run_inference_pipeline
    except ImportError as e:
        print(f"Error importing hailo modules: {e}")
        return
    
    # Initialize database and set mavlink master for telemetry
    init_database()
    set_mavlink_master(master)
    set_camera_config(camera_config)
    
    print("\n📷 Starting Object Detection with Geotagging...")
    
    try:
        # Run inference pipeline with tracking enabled
        run_inference_pipeline(
            net=hef_path,
            input_source=input_source,
            track=True,
            show=False  # Set to True if you want live display
        )
    except Exception as e:
        print(f"Object detection error: {e}")
    
    print("📷 Object Detection stopped")


def monitor_mission(master):
    """
    Monitor mission progress and print status.
    
    Args:
        master: pymavlink connection.
    """
    global _stop_inference
    
    print("\n📡 Monitoring mission progress...")
    
    while not _stop_inference:
        # Check mission status
        msg = master.recv_match(type=["MISSION_CURRENT", "HEARTBEAT"], blocking=True, timeout=2)
        
        if msg:
            if msg.get_type() == "MISSION_CURRENT":
                print(f"  Current waypoint: {msg.seq}")
            elif msg.get_type() == "HEARTBEAT":
                # Check if mission complete (RTL or landed)
                mode = mavutil.mode_string_v10(msg)
                if mode in ["RTL", "LAND"]:
                    print(f"  Mode changed to {mode} - Mission ending")
                    break
        
        # Get current position
        pos = get_current_position(master, config)
        if pos:
            lat, lon, alt = pos
            print(f"  Position: ({lat:.6f}, {lon:.6f}) Alt: {alt:.1f}m")
        
        # Get attitude
        att = get_attitude(master, config)
        if att:
            roll, pitch, yaw = att
            print(f"  Attitude: Roll={roll:.1f}° Pitch={pitch:.1f}° Yaw={yaw:.1f}°")
        
        time.sleep(2)
    
    _stop_inference = True


def print_tracklet_summary():
    """Print summary of detected tracklets from database."""
    try:
        from src.hailo.object_detection.tracklet_database import get_all_tracklets
        
        tracklets = get_all_tracklets()
        
        print("\n" + "="*60)
        print("📍 DETECTED TRACKLETS SUMMARY")
        print("="*60)
        
        if not tracklets:
            print("No tracklets detected during mission.")
            return
        
        print(f"Total unique tracks: {len(tracklets)}")
        print("-"*60)
        
        for t in tracklets:
            print(f"  Track {t['track_id']:3d} | {t['class_name']:10s} | "
                  f"GPS: ({t['latitude']:.6f}, {t['longitude']:.6f}) | "
                  f"Conf: {t['confidence']*100:.1f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error printing tracklet summary: {e}")


# ======================================================
# MAIN FLOW
# ======================================================
def main():
    global _stop_inference
    
    # Load camera config
    camera_config = {}
    if os.path.exists(camera_config_path):
        with open(camera_config_path, 'r') as f:
            camera_config = json_load(f)
        print(f"✔ Loaded camera config from {camera_config_path}")
    else:
        print(f"⚠ Camera config not found at {camera_config_path}")
        print("  GPS projection will not work without camera intrinsics")
    
    # Connect to vehicle
    master = connect_vehicle(config)
    
    # Upload mission
    upload_mission(master, WAYPOINTS)
    
    # Pre-arm checks
    gps_prearm_check(master, config)
    
    # Arm and takeoff
    set_mode(master, config, "GUIDED")
    arm(master, config)
    takeoff(master, config, config.get("TAKEOFF_ALT", 20))
    
    # Start AUTO mission
    set_mode(master, config, "AUTO")
    print("\n🚀 Mission Running — RTL will execute automatically at end\n")
    
    # Configuration for object detection
    # Adjust these paths for your setup
    hef_path = config.get("HEF_MODEL_PATH", os.path.expanduser("~/hailo-apps/yolo8s.hef"))
    input_source = config.get("CAMERA_SOURCE", "0")  # "0" for camera, or video file path
    
    # Start object detection in separate thread
    detection_thread = threading.Thread(
        target=run_object_detection,
        args=(master, camera_config, hef_path, input_source),
        daemon=True
    )
    detection_thread.start()
    
    # Monitor mission in main thread
    try:
        monitor_mission(master)
    except KeyboardInterrupt:
        print("\n⚠ Mission interrupted by user")
    finally:
        _stop_inference = True
    
    # Wait for detection thread to finish
    detection_thread.join(timeout=5)
    
    # Print summary of detected tracklets
    print_tracklet_summary()
    
    print("\n✅ Mission Complete")


if __name__ == "__main__":
    main()