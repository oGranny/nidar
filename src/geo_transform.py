import math
import numpy as np


def person_gps_from_bbox(
    bbox_x1,
    bbox_y1,
    bbox_x2,
    bbox_y2,
    drone_lat,
    drone_lon,
    altitude_m,
    heading_deg,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    fx=920.0,
    fy=920.0,
    cx=640.0,
    cy=360.0,
):
    """
    Estimate the ground GPS position of a detected person using the bounding box center.
    Returns a tuple of (person_latitude, person_longitude) or None if the viewing ray misses the ground plane.
    """

    # 1. Bounding box center in pixel space
    u = (bbox_x1 + bbox_x2) / 2.0
    v = (bbox_y1 + bbox_y2) / 2.0

    # 2. Convert pixel coordinates to a normalized camera ray
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    ray_cam = np.array([x_cam, y_cam, 1.0])

    # 3. Camera orientation matrices
    def rx(angle):
        return np.array(
            [
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)],
            ]
        )

    def ry(angle):
        return np.array(
            [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ]
        )

    def rz(angle):
        return np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    heading_rad = math.radians(heading_deg)

    rotation = rz(heading_rad) @ rz(yaw) @ ry(pitch) @ rx(roll)
    ray_world = rotation @ ray_cam

    # 4. Intersect the ray with the ground plane (z = 0)
    if ray_world[2] >= 0:
        return None

    t = altitude_m / (-ray_world[2])
    east_m = t * ray_world[0]
    north_m = t * ray_world[1]

    # 5. Convert local ENU meters to GPS deltas
    meters_per_deg_lat = 111111.0
    meters_per_deg_lon = 111111.0 * math.cos(math.radians(drone_lat))

    person_lat = drone_lat + (north_m / meters_per_deg_lat)
    person_lon = drone_lon + (east_m / meters_per_deg_lon)

    return person_lat, person_lon
