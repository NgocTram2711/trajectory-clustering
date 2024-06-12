from xml.etree import ElementTree


def contains_gpx_data(file_path):
    NSMAP = {"gpx": "http://www.topografix.com/GPX/1/0"}
    try:
        tree = ElementTree.parse(file_path)
        root = tree.getroot()
        return bool(root.find(".//gpx:trk", namespaces=NSMAP))
    except ElementTree.ParseError:
        return False


def find_closest_segment(gdf, point):
    distances = gdf["geometry"].apply(lambda flow: flow.distance(point))
    return gdf.at[distances.idxmin(), "weight"]
