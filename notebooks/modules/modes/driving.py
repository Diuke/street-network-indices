import geopandas as geopd
import pandas as pd
import numpy as np
import osmnx as ox

# Approx conversion of Degrees to Meters
# 1 deg = 111000 m
# 0.1 deg = 11100.0 m
# 0.01 deg = 1110.00 m
# 0.001 deg = 111.000 m
# 0.0001 deg = 11.1000 m
# 0.00001 deg = 1.11000 m
DIST_THRESHOLD = 20 # meters
SLOPE_THRESHOLD = 15 # 5 degree (angle) similarity
SIDEWALK_GRAPH_DEPTH = 5 # Induced subgraph of 10 nodes deep

DRIVING_FIELDS = ["length","oneway","lanes","ref","name","highway","maxspeed","service","width","junction"]

# custom cycling filter based on the overpass bike filter
CUSTOM_DRIVING_FILTER = (
    f'["highway"]["area"!~"yes"]'
    f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
    f"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|steps|"
    f'track"]'
    f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
    f'["service"!~"emergency_access|parking|parking_aisle|private"]'
)

def process_driving_graph(gdf: geopd.GeoDataFrame):
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
    for field in DRIVING_FIELDS:
        if field not in edge_cols:
            edges_g[field] = None

    # keep only the necessary columns
    edges_g = edges_g[["u","v","key","geometry"] + DRIVING_FIELDS]

    # Apply driving filters
    edge_cols = list(edges_g.columns)
    # Remove streets with no access
    if "access" in edge_cols:
        edges_g = edges_g.loc[edges_g["access"] != "no"]

    # reset the edges u,v,key index for rebuilding the graph.
    edges_g = edges_g.set_index(['u', 'v', 'key'])

    # reset the node index (osmid) for rebuilding the graph.
    nodes_g = nodes_g.drop(["index"], axis=1)
    nodes_g.index = nodes_g["osmid"]
    try:
        nodes_g = nodes_g.drop(["osmid"])
    except:
        pass

    # simplify to remove intestitial nodes
    print("start rebuilding graph")
    remade = ox.graph_from_gdfs(nodes_g, edges_g)
    print("finish rebuilding graph")
    print("start graph simplify")
    remade = ox.simplify_graph(remade, edge_attrs_differ=["highway"])
    print("finish graph simplify")
    new_g = ox.add_edge_bearings(remade)

    # add speeds and travel times
    new_g = ox.add_edge_speeds(new_g, fallback=30) #TODO CHANGE THE DEFAULT SPEED
    new_g = ox.add_edge_travel_times(new_g)

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