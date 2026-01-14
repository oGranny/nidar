import os
from flask import json
from pymavlink import mavutil
import time

from helper.mavlink import arm, connect_vehicle, gps_prearm_check, set_mode, takeoff, upload_mission

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

WAYPOINTS = [
    (12.9715987, 77.5945660, 20),
    (12.9717000, 77.5948000, 25),
    (12.9719000, 77.5950000, 30),
]


# ======================================================
# MAIN FLOW
# ======================================================
def main():
    master = connect_vehicle(config)
    upload_mission(master, WAYPOINTS)
    gps_prearm_check(master, config)
    set_mode(master, config, "GUIDED")
    arm(master, config)
    takeoff(master, config, config.get("TAKEOFF_ALT"))
    set_mode(master, config, "AUTO")
    print("\n🚀 Mission Running — RTL will execute automatically at end\n")

    # while True:
    #     msg = master.recv_match(type=["MISSION_CURRENT", "NAV_CONTROLLER_OUTPUT"], timeout=2)
    #     if msg:   
    #         print(msg)


if __name__ == "__main__":
    main()