import requests
import geopandas as geopd
import numpy as np
import osmnx as ox
import networkx as nx
import modules.utils as utils
import shapely
from osmnx import settings

# Public transport in OSM
# Nodes: public_transport = stop_position
# Edges

# according to https://wiki.openstreetmap.org/wiki/Guidelines_for_pedestrian_navigation
PUBLIC_TRANSPORT_NODE_FIELDS = [
    "public_transport", "name", "ref", "uic_ref", "uicname", "operator", "network"
]
PUBLIC_TRANSPORT_EDGE_FIELDS = ["length","name","highway","access","sidewalk","foot","footway"]

 # custom walk filter based on the overpass walk filter
CUSTOM_PUBLIC_TRANSPORT_FILTER = (
    f'relation["route"~"train|subway|monorail|tram|bus|trolleybus|aerialway|ferry"]["public_transport"!="stop_area"]]'
)

def process_public_transport_graph(geom: shapely.Polygon):

    # Obtain public transport relations for the polygon from Overpass
    # Use the OSMnx overpass module for building the overpass geometry
    aoi = ox._overpass._make_overpass_polygon_coord_strs(geom)[0] 
    # Build the Overpass query
    overpass_query = f'[out:json][timeout:60];(rel(poly:{aoi!r})[\"route\"~\"train|subway|monorail|tram|bus|trolleybus|ferry\"];rel(r)(poly:{aoi!r}););out;'
    # use the OSMnx overpass url
    overpass_interpreter_url = settings.overpass_url + "/interpreter" #"https://overpass-api.de/api/interpreter"
    body = {
        "data": overpass_query
    }
    r = requests.post(overpass_interpreter_url, body)
    relations = r.json()["elements"]

    # Obtain public transport nodes (stop position and platforms) from OpenStreetMap using OSMnx.
    tags = {'public_transport': ['stop_position','platform']}
    gdfox = ox.features_from_polygon(geom, tags)
    gdfox = gdfox.reset_index()
    gdfox = gdfox.loc[gdfox["element"] == "node"]
    gdfox = gdfox.set_index("id")
    # Build a hashmap based on the osmid for faster access
    nodes_dict = gdfox.to_dict(orient="index")

    # Build the graph with the relationships and nodes downloaded
    public_graph = nx.MultiDiGraph()

    # Add all the nodes to the graph first
    for n,node_data in nodes_dict.items():
        data = {
            "name": node_data['name'] if "name" in node_data else None,
            "x": node_data['geometry'].x,
            "y": node_data['geometry'].y,

            "train": node_data['train'] if "train" in node_data else None,
            "subway": node_data['subway'] if "subway" in node_data else None,
            "monorail": node_data['monorail'] if "monorail" in node_data else None,
            "tram": node_data['tram'] if "tram" in node_data else None,
            "bus": "yes" if ("bus" in node_data and node_data["bus"] == "yes") or ("highway" in node_data and node_data["highway"] == "bus_stop") else None,
            "trolleybus": node_data['trolleybus'] if "trolleybus" in node_data else None,
            "ferry": node_data['ferry'] if "ferry" in node_data else None,
        }
        public_graph.add_node(n, **data)

    # Add edges from the relations.
    # The OSM relation gives in order the nodes composing each route. 
    for rel in relations:
        rel_metadata = rel["tags"]
        node_list = list(filter(lambda x: x["type"] != "way" and (x["role"] == "stop" or x["role"] == "platform"), rel["members"]))
        
        prev_stop = None
        for node in node_list:
            ref = node["ref"]
            if ref in nodes_dict:
                if prev_stop is None:
                    # first node
                    prev_stop = ref
                else:
                    edge_data = {
                        "geometry": shapely.LineString([nodes_dict[prev_stop]["geometry"], nodes_dict[ref]["geometry"]]),
                        "description": rel_metadata['description'] if "description" in rel_metadata else None,
                        "from": rel_metadata['from'] if "from" in rel_metadata else None,
                        "to": rel_metadata['to'] if "to" in rel_metadata else None,
                        "operator": rel_metadata['operator'] if "operator" in rel_metadata else None,
                        "network": rel_metadata['network'] if "network" in rel_metadata else None,
                        "type": rel_metadata['type'] if "type" in rel_metadata else None,
                        "route": rel_metadata['route'] if "route" in rel_metadata else None
                    }
                    u = prev_stop
                    v = ref
                    public_graph.add_edge(u,v, **edge_data)
                    prev_stop = ref

            else:
                prev_stop = None

    public_graph.graph["crs"] = "epsg:4326"

    return public_graph