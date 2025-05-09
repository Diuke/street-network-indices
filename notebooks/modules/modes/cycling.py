import geopandas as geopd
import numpy as np
import osmnx as ox
import modules.utils as utils

CYCLING_GRAPH_EDGE_TAGS = [
    "oneway",
    "lanes",
    "ref",
    "name",
    "highway",
    "maxspeed",
    "service",
    "footway",
    "access",
    "width",
    "est_width",
    "junction",
    "sidewalk",
    "footway",
    "foot",
    "bike",
    "bicycle",
    "area",
    "sidewalk:both",
    "sidewalk:left",
    "sidewalk:right",
    "cycleway",
    "cycling_width"
]
CYCLING_GRAPH_NODE_TAGS = []

CYCLING_FIELDS = ["length","name","highway","access","lanes","bicycle","lanes","cycleway"]

 # custom cycling filter based on the overpass bike filter
CUSTOM_CYCLING_FILTER = (
    f'["highway"]["area"!~"yes"]["access"!~"private"]'
    f'["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator|'
    f'motor|no|planned|platform|proposed|raceway|razed|steps"]'
    f'["bicycle"!~"no"]["service"!~"private"]'
    f'["cycleway"!~"separate"]["cycleway:both"!~"separate"]'
    f'["cycleway:left"!~"separate"]["cycleway:right"!~"separate"]'
)

AGGREGATION_FUNCTIONS = {
    "length": "sum",
    "name": "sum",
    "highway": "",
    "access": "",
    "lanes": "",
    "bicycle": "",
    "lanes": "",
    "cycleway": ""
}

def process_cycling_graph(gdf: geopd.GeoDataFrame, assessment=False):
    """
    Directed graph
    """
    # Reset index twice to remove the u,v,k relationship and assign a unique index to
    # each node and edge
    nodes_g = gdf[0].reset_index().reset_index()
    edges_g = gdf[1].reset_index().reset_index()

    nodes_g = nodes_g.replace(np.nan, None)
    edges_g = edges_g.replace(np.nan, None)

    # Create columns if they do not exist
    edge_cols = list(edges_g.columns)
    for field in CYCLING_FIELDS:
        if field not in edge_cols:
            edges_g[field] = None

    # keep only the necessary columns
    edges_g = edges_g[["u","v","key","geometry"] + CYCLING_FIELDS]

    # Apply cycling filters
    edge_cols = list(edges_g.columns)

    # Remove streets with no access
    edges_g = edges_g.loc[edges_g["access"] != "no"]

    # Keep segments with bicycle values that specify that cycling is allowed or undefined (None).
    edges_g = edges_g.loc[
        (edges_g["bicycle"].isin(["designated", "yes", "discouraged", "optional_sidepath", None]))
    ]

    # keep footways only if they specify that bicycle is allowed
    edges_g = edges_g.loc[
        ~(
            (edges_g["highway"] == "footway") &
            ~(edges_g["bicycle"].isin(["designated", "yes", "discouraged", "optional_sidepath"]))
        )
    ]        

    # reset the edges u,v,key index for rebuilding the graph.
    edges_g = edges_g.set_index(['u', 'v', 'key'])

    # reset the node index (osmid) for rebuilding the graph.
    nodes_g = nodes_g.drop(["index"], axis=1)
    nodes_g.index = nodes_g["osmid"]
    try:
        nodes_g = nodes_g.drop(["osmid"])
    except:
        pass

    # add speed and travel times based on average cycling speed of 15 km/h
    edges_g["speed_kph"] = 15 # average cycling speed of 15km/h
    # convert distance meters to km, and speed km per hour to km per second
    distance_km = edges_g["length"] / 1000
    speed_km_sec = edges_g["speed_kph"] / (60 * 60)
    # calculate edge travel time in seconds
    travel_time = distance_km / speed_km_sec
    edges_g["travel_time"] = travel_time 

    # simplify to remove intestitial nodes
    print("start rebuilding graph")
    remade = ox.graph_from_gdfs(nodes_g, edges_g)
    print("finish rebuilding graph")
    print("start graph simplify")
    remade = ox.simplify_graph(remade, edge_attrs_differ=["bicycle", "highway", "cycleway"])
    print("finish graph simplify")
    new_g = ox.add_edge_bearings(remade)

    # remove single disconnected edges 
    remove_singles = []
    for u,v,key in new_g.edges(data=False, keys=True):
        u_pred = len(list(new_g.predecessors(u)))
        u_succ = len(new_g[u])

        v_pred = len(list(new_g.predecessors(v)))
        v_succ = len(new_g[v])

        if ((u_pred <= 1 and u_succ <= 1) and (v_pred <= 1 and v_succ <= 1)):
            remove_singles.append([u,v,key])
    new_g.remove_edges_from(remove_singles)
    
    # remove single disconnected nodes 
    remove_singles = []
    for node_id in new_g.nodes(data=False):
        if len(new_g[node_id]) == 0:
            remove_singles.append(node_id)
    new_g.remove_nodes_from(remove_singles)
    
    # return the processed graph
    return new_g