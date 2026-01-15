import os
import math
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
METERS_PER_DEG_LAT = 111000.0  # local scale used elsewhere
BUFFER_KML = 10.0 # buffer in meters

def parse_kml_polygon_coords(kml_path):
    """Extract (lon, lat) coordinates from the first Polygon->LinearRing in the KML."""
    if not os.path.exists(kml_path):
        raise FileNotFoundError(kml_path)

    tree = ET.parse(kml_path)
    root = tree.getroot()

    coords_elem = root.find(
        ".//kml:Polygon//kml:LinearRing//kml:coordinates",
        namespaces=KML_NS,
    )
    if coords_elem is None or not coords_elem.text:
        raise ValueError("No Polygon coordinates found in KML.")

    raw = coords_elem.text.strip()
    points = []
    for triplet in raw.split():
        parts = triplet.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 else 0.0
        points.append((lon, lat, alt))

    if len(points) < 3:
        raise ValueError("Not enough points to form a polygon.")

    return points

def _segment_length_m(p0, p1):
    """Approximate distance between two lon/lat points in meters (local scale)."""
    x0, y0 = p0
    x1, y1 = p1
    dx = (x1 - x0) * METERS_PER_DEG_LAT * math.cos(math.radians((y0 + y1) / 2.0))
    dy = (y1 - y0) * METERS_PER_DEG_LAT
    return math.hypot(dx, dy)

def _interpolate_by_distance(p0, p1, max_seg_m):
    """
    Interpolate points between p0 and p1 so that no segment is longer than max_seg_m.
    Returns [p0, ..., p1].
    """
    if max_seg_m <= 0:
        return [p0, p1]

    dist = _segment_length_m(p0, p1)
    if dist <= max_seg_m:
        return [p0, p1]

    n = int(math.ceil(dist / max_seg_m))  # number of subsegments
    step = 1.0 / n

    x0, y0 = p0
    x1, y1 = p1
    pts = [p0]
    for i in range(1, n):
        t = i * step
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        pts.append((x, y))
    pts.append(p1)
    return pts

def generate_parallel_path(border_coords, spacing_m, angle_deg,  max_seg_m=0.0, alt=20,):
    """
    Generate a lawnmower/parallel path inside the polygon defined by border_coords.

    border_coords: list[(lon, lat)]
    spacing_m: spacing between lines in meters
    angle_deg: angle of parallel lines
    max_seg_m: max segment length in meters between consecutive waypoints.
               0 or less = no extra interpolation.
    """
    poly = Polygon(border_coords)

    if BUFFER_KML != 0:
        poly = poly.buffer(-BUFFER_KML / METERS_PER_DEG_LAT)
        if poly.is_empty:
            return []

    spacing_deg = spacing_m / METERS_PER_DEG_LAT
    rotated = rotate(poly, -angle_deg, origin="centroid")
    min_x, min_y, max_x, max_y = rotated.bounds

    lines = []
    y = min_y
    while y <= max_y:
        base_line = LineString([(min_x, y), (max_x, y)])
        inter = base_line.intersection(rotated)
        if not inter.is_empty:
            if inter.geom_type == "LineString":
                lines.append(inter)
            elif inter.geom_type == "MultiLineString":
                lines.extend(inter.geoms)
        y += spacing_deg

    coarse = []
    for i, ln in enumerate(lines):
        world_ln = rotate(ln, angle_deg, origin=poly.centroid)
        pts = list(world_ln.coords)
        if i % 2 == 1:  # lawnmower pattern
            pts.reverse()
        coarse.extend(pts)

    # Interpolate based on distance
    if max_seg_m > 0 and len(coarse) > 1:
        dense = []
        for i in range(len(coarse) - 1):
            p0 = (coarse[i][0], coarse[i][1])
            p1 = (coarse[i + 1][0], coarse[i + 1][1])
            seg_pts = _interpolate_by_distance(p0, p1, max_seg_m)
            if i > 0:
                seg_pts = seg_pts[1:]  # avoid duplicates
            dense.extend(seg_pts)
        waypoints = dense
    else:
        waypoints = [(x, y) for x, y in coarse]

    return [(lon, lat, alt) for lon, lat in waypoints]

def main():
    kml_path = r"f:/nidar/dts.kml"

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

    waypoints = generate_parallel_path(
        border,
        spacing_meters,
        angle_degrees,
        max_seg_m=max_seg_m,
    )
    print(f"Generated {len(waypoints)} waypoints (max_seg_m = {max_seg_m} m).")
    print("waypoints = [")
    for lon, lat, alt in waypoints:
        print(f"    ({lon:.12f}, {lat:.12f}, {alt}),")
    print("]")

if __name__ == "__main__":
    main()