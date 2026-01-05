import os
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET

KML_NS = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NS)


def create_kml_document(
    name: str,
    waypoints: Iterable[Tuple[float, float]],
    line_color: str = "ff0000ff",  # aabbggrr (red here)
    line_width: float = 2.0,
    point_icon_scale: float = 0.8,
) -> ET.ElementTree:
    """
    Create an ElementTree KML document with:
      - a LineString connecting all waypoints
      - individual Placemarks for each waypoint

    waypoints: iterable of (lon, lat) in decimal degrees.
    line_color: KML ABGR hex color (e.g., "ff0000ff" for red).
    """

    kml = ET.Element(ET.QName(KML_NS, "kml"))
    document = ET.SubElement(kml, ET.QName(KML_NS, "Document"))

    name_elem = ET.SubElement(document, ET.QName(KML_NS, "name"))
    name_elem.text = name

    # Styles
    style = ET.SubElement(document, ET.QName(KML_NS, "Style"), {"id": "waypointsStyle"})

    icon_style = ET.SubElement(style, ET.QName(KML_NS, "IconStyle"))
    scale_elem = ET.SubElement(icon_style, ET.QName(KML_NS, "scale"))
    scale_elem.text = str(point_icon_scale)
    icon = ET.SubElement(icon_style, ET.QName(KML_NS, "Icon"))
    href = ET.SubElement(icon, ET.QName(KML_NS, "href"))
    href.text = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"

    line_style = ET.SubElement(style, ET.QName(KML_NS, "LineStyle"))
    color_elem = ET.SubElement(line_style, ET.QName(KML_NS, "color"))
    color_elem.text = line_color
    width_elem = ET.SubElement(line_style, ET.QName(KML_NS, "width"))
    width_elem.text = str(line_width)

    coords_str = " ".join(f"{lon:.12f},{lat:.12f},0" for lon, lat in waypoints)
    if not coords_str:
        raise ValueError("No waypoints provided to create KML.")

    # LineString Placemark
    line_pm = ET.SubElement(document, ET.QName(KML_NS, "Placemark"))
    line_name = ET.SubElement(line_pm, ET.QName(KML_NS, "name"))
    line_name.text = f"{name} - Path"
    style_url = ET.SubElement(line_pm, ET.QName(KML_NS, "styleUrl"))
    style_url.text = "#waypointsStyle"

    line_string = ET.SubElement(line_pm, ET.QName(KML_NS, "LineString"))
    tessellate = ET.SubElement(line_string, ET.QName(KML_NS, "tessellate"))
    tessellate.text = "1"
    coords = ET.SubElement(line_string, ET.QName(KML_NS, "coordinates"))
    coords.text = coords_str

    # Individual point Placemarks
    for idx, (lon, lat) in enumerate(waypoints, start=1):
        pm = ET.SubElement(document, ET.QName(KML_NS, "Placemark"))
        p_name = ET.SubElement(pm, ET.QName(KML_NS, "name"))
        p_name.text = f"WP {idx}"
        p_style = ET.SubElement(pm, ET.QName(KML_NS, "styleUrl"))
        p_style.text = "#waypointsStyle"

        point = ET.SubElement(pm, ET.QName(KML_NS, "Point"))
        p_coords = ET.SubElement(point, ET.QName(KML_NS, "coordinates"))
        p_coords.text = f"{lon:.12f},{lat:.12f},0"

    return ET.ElementTree(kml)


def save_waypoints_to_kml(
    waypoints: Iterable[Tuple[float, float]],
    output_path: str,
    name: str = "Generated Waypoints",
    line_color: str = "ff0000ff",
    line_width: float = 2.0,
    point_icon_scale: float = 0.8,
) -> str:
    """
    Create and save a KML file with the given waypoints.

    Returns the absolute path of the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tree = create_kml_document(
        name=name,
        waypoints=waypoints,
        line_color=line_color,
        line_width=line_width,
        point_icon_scale=point_icon_scale,
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    # Example usage / manual test
    example_waypoints = [
    (84.907887989113, 22.254483479856),
    (84.907887989113, 22.254813049318),
    (84.907887989113, 22.255142618781),
    (84.907887989113, 22.255472188243),
    (84.907707808932, 22.255601056082),
    (84.907707808932, 22.255277013079),
    (84.907707808932, 22.254952970077),
    (84.907707808932, 22.254628927075),
    (84.907707808932, 22.254304884073),
    (84.907527628752, 22.254201693320),
    (84.907527628752, 22.254497199807),
    (84.907527628752, 22.254792706294),
    (84.907527628752, 22.255088212781),
    (84.907527628752, 22.255383719268),
    (84.907527628752, 22.255679225755),
    (84.907347448572, 22.255736538190),
    (84.907347448572, 22.255425556002),
    (84.907347448572, 22.255114573813),
    (84.907347448572, 22.254803591625),
    (84.907347448572, 22.254492609437),
    (84.907347448572, 22.254181627248),
    (84.907167268392, 22.254225128973),
    (84.907167268392, 22.254526817003),
    (84.907167268392, 22.254828505033),
    (84.907167268392, 22.255130193063),
    (84.907167268392, 22.255431881093),
    (84.907167268392, 22.255733569123),
    (84.906987088212, 22.255706336887),
    (84.906987088212, 22.255409558527),
    (84.906987088212, 22.255112780168),
    (84.906987088212, 22.254816001809),
    (84.906987088212, 22.254519223449),
    (84.906987088212, 22.254222445090),
    (84.906806908032, 22.254253709668),
    (84.906806908032, 22.254605570442),
    (84.906806908032, 22.254957431216),
    (84.906806908032, 22.255309291990),
    (84.906806908032, 22.255661152764),
    (84.906626727851, 22.255541469373),
    (84.906626727851, 22.255229725833),
    (84.906626727851, 22.254917982293),
    (84.906626727851, 22.254606238753),
    (84.906626727851, 22.254294495212),
    (84.906536637761, 22.254564608968),
    (84.906446547671, 22.254834722724),
    (84.906446547671, 22.255064735844),
    (84.906446547671, 22.255294748963),
    (84.906266367491, 22.255057983192),
    (84.906266367491, 22.254813078057),    ]    
    out_file = save_waypoints_to_kml(
        example_waypoints,
        "waypoints_output.kml",
        name="Example Waypoints",
    )
    print(f"KML written to: {out_file}")