import os
import sys
import time
import subprocess
import json

from pymavlink import mavutil

# Ensure imports work when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.helper.mavlink import arm, connect_vehicle, gps_prearm_check, set_mode, takeoff, upload_mission
from src.navigation.extract_waypoint import  generate_parallel_path, parse_kml_polygon_coords
from src.navigation.geofence import upload_geofence, kml_to_boundary_coordinates

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

def extract_kml(kml_path):

    spacing_meters = 20.0   # distance between parallel lines
    angle_degrees = 90.0     # orientation of parallel lines
    max_seg_m = 40.0         # max distance between consecutive waypoints

    print(f"Reading KML: {kml_path}")
    border = parse_kml_polygon_coords(kml_path)

    print(f"\nBorder has {len(border)} points.")
    print("border_coords = [")
    for lon, lat in border:
        print(f"    ({lon:.12f}, {lat:.12f}),")
    print("]\n")

    WAYPOINTS = generate_parallel_path(
        border,
        spacing_meters,
        angle_degrees,
        max_seg_m=max_seg_m,
    )
    print(f"Generated {len(WAYPOINTS)} waypoints (max_seg_m = {max_seg_m} m).")
    print(WAYPOINTS)
    return WAYPOINTS






def start_hailo_pipeline():
    """
    Runs: cd src/hailo/object_detection/ && python object_detection.py -n ... -i ... --track
    """
    # Read from config.json if present, otherwise use defaults
    net_path = config.get("HEF_MODEL_PATH", os.path.expanduser("~/hailo-apps/yolo8s.hef"))
    input_src = config.get("DETECTION_INPUT", os.path.expanduser("~/hailo-apps/videop401231.mp4"))

    cmd = [
        "python",
        "object_detection.py",  # <-- Fixed: added missing comma
        "-n", net_path,
        "-i", 'camera',
        "--track",
    ]

    # Run from the object_detection folder (same as running from terminal)
    cwd = os.path.join(os.path.dirname(__file__), "src", "hailo", "object_detection")
    
    print(f"Starting Hailo pipeline: {' '.join(cmd)} (cwd={cwd})")

    # Blocking run (waits until inference completes)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


# ======================================================
# MAIN FLOW
# ======================================================
def main():
    master = connect_vehicle(config)
    WAYPOINTS = extract_kml(config.get("KML_PATH", "./src/navigation/dts.kml"))
    geof = kml_to_boundary_coordinates(config.get("KML_PATH", "./src/navigation/dts.kml"))
    print(geof)
    upload_geofence(master, geof)
    print()
    # upload_mission(master, WAYPOINTS)

    # gps_prearm_check(master, config)
    set_mode(master, config, "GUIDED")
    # arm(master, config)
    # takeoff(master, config, config.get("TAKEOFF_ALT"))
    set_mode(master, config, "AUTO")
    print("\nMission Running — RTL will execute automatically at end\n")

    # Start Hailo pipeline after mission starts
    rc = start_hailo_pipeline()
    print(f"Hailo pipeline exited with code: {rc}")


if __name__ == "__main__":
    main()