import copy
import math
import ray
import geopandas as geopd
import numpy as np
import osmnx as ox
import networkx as nx
import modules.utils as utils

DIST_THRESHOLD = 20 # meters
"""Default distance that a parallel street must be from a sidewalk."""

SLOPE_THRESHOLD = 15 # 5 degree (angle) similarity
"""Default angle difference for sidewalks and streets to be parallel."""

SIDEWALK_GRAPH_DEPTH = 5 # Induced subgraph of 10 nodes deep
"""Default BFS graph depth to search for parallel streets and sidewalks."""

PEDESTRIAN_GRAPH_EDGE_TAGS = [
    "oneway",
    "lanes",
    "ref",
    "name",
    "highway",
    "service",
    "footway",
    "access",
    "width",
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
"""Edge tags to dowload for pedestrian network."""

PEDESTRIAN_GRAPH_NODE_TAGS = []
"""Node tags to dowload for pedestrian network."""

# according to https://wiki.openstreetmap.org/wiki/Guidelines_for_pedestrian_navigation
PEDESTRIAN_TYPES = ["footway","pedestrian","living_street","path","track","steps","cycleway"]
"""Types of pedestrian streets in OSM. Extracted from the guidelines for pedestrian navigation."""

PEDESTRIAN_FIELDS = ["length","name","highway","access","sidewalk","foot","footway"]
"""Useful fields for pedestrian networks."""

 # custom walk filter based on the overpass walk filter
CUSTOM_WALK_FILTER = (
    f'["highway"]["area"!~"yes"]["access"!~"private"]' # remove areas and private access places
    f'["highway"!~"abandoned|bus_guideway|construction|motor|no|planned|platform|proposed|raceway|razed"]' # removing non pedestrian highway types
    f'["foot"!~"no"]' # removing where foot is not allowed
    f'["service"!~"private"]' # removing private service streets
    f'["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]' # removing streets with separately mapped sidewalks
    f'["level"!~"."]' # removing indoor walking paths
)
"""
Custom Overpass API filter for pedestrian networks.

Built upon the OSMnx walk filter, adds additional filters for removing streets with explicitly specified sidewalks and additional restrictions.
"""

def _pedestrian_remove_sidewalks(
        street_types:list[str], 
        process_edges:list[dict["u":any,"v":any,"key":any,"data":any]], 
        original_graph:nx.MultiGraph, 
        nodes_gdf:geopd.GeoDataFrame,
        edges_gdf:geopd.GeoDataFrame,
        dist_threshold:int,
        slope_threshold:int,
        assess=False,
    ) -> list[dict["u":any,"v":any,"key":any,"assess":bool]]:
    """
    Remove streets if they are already covered by sidewalks and OSM does not explicitly speficy its existance (sidewalk=separate or sidewalk:sidewalk are not present).
    
    This algorithm visits each street segment (edge of the graph) and analyses the surrounding subgraph in search of a pedestrian sidewalk that matches the angle of the street and that is close enough to be considered the same street.

    Although this algorithm is not 100% accurate, it removes most of the streets that contain sidewalks. If the assess parameter is sent, 
    instead of removing the duplicated streets, it will set a new property "assess" that tells reviewers that the street may need assessment.

    Parameters
    ----------
    street_types (list[str]): 
        List of the OSM street types (highway tag values) to review. E.g., primary, secondary, residential, etc.

    process_edges (list[dict["u":any,"v":any,"key":any,"data":any]]): 
        A list of edges [u,v,key,data] to process. This allows to calculate only a subset of the graph for speeding up computations, as
        usually parallel sidewalks are within a few street segments from the street they represent.

    original_graph (networkx.MultiGraph): 
        The networkX graph containing the complete graph to analyse. This graph edges' must have the properties "highway", "bearing", and "sidewalk"
        for checking if a sidewalks is parallel to a street. 

    slope_threshold (int): 
        The maximum slope (bearing difference) to consider 2 segments parallel. 

    dist_threshold (int): 
        The maximum distance in meters to consider a sidewalk and street as duplicated.

    Returns
    -------
    A sublist of edges [u,v,key] from the process_edges to keep on the graph. Streets with sidewalks are removed. 

    """
    #approximately convert to degrees -> 1 deg = 111595.75 meters
    # sum 5 meters for the erosion
    DEG_CONVERT = 111595.75 
    buffer_distance = dist_threshold / DEG_CONVERT
    # 5 meters erosion
    erode_distance = 5 / DEG_CONVERT

    # save the edges to keep
    keep_edges = []
    print(f"starting with {len(process_edges)} edges")
    for u,v,key,data in process_edges:        
        # if for some reason the data of the edge is empty, continue
        remove = False
        try:
            if not data: continue
        except Exception as ex:
            print(f"error - skip {u},{v}")

        # If the edge has not been yet removed:
        if original_graph.has_edge(u,v,key):
            do_assessment = False       
            # check if the type of street is one of the types that are removable
            # if the street is composed by many, the highway type is merged.
            if '[' in data["highway"]:
                separated = str(data["highway"]).replace("[","").replace("]","").replace("'","").split(",")
                is_street = False
                for s in separated:
                    if s in street_types:
                        is_street = True
            else:
                is_street = data["highway"] in street_types

            if is_street:
                # if it is a self loop (bearing undefined) - keep and skip
                m = data["bearing"]
                if m is None:
                    keep_edges.append([u,v,key,do_assessment])
                    continue
                
                #if the street is too small and the erosion is larger than the street continue - keep and skip
                if data["geometry"].length  <= (erode_distance * 2):
                    keep_edges.append([u,v,key,do_assessment])
                    continue

                # meters to degree approximate conversion
                street_buffer = utils.buffer(data["geometry"], buffer_distance, erode_distance)

                # create a subgraph with the induced nodes from the U and V nodes of the edge with a depth of sidewalk_graph_depth
                nearest_idx = edges_gdf.sindex.query(street_buffer, predicate="intersects")
                edges_in_buffered_graph = edges_gdf.iloc[nearest_idx]

                subgraph = original_graph.edge_subgraph(list(edges_in_buffered_graph.index))
                #subgraph = utils.explore_graph(u, original_graph, SIDEWALK_GRAPH_DEPTH)

                # Iterate the nodes of the subgraph
                for sub_e_u, sub_e_v in subgraph.edges():
                    # The nodes are integers, but come as strings
                    sub_e_u = int(sub_e_u)
                    sub_e_v = int(sub_e_v)

                    if u == sub_e_u and v == sub_e_v: continue
                    if v == sub_e_u and u == sub_e_v: continue
                    if v == sub_e_u and u == sub_e_v: continue

                    multi_edges_data = original_graph.get_edge_data(sub_e_u, sub_e_v, default=None)
                    if multi_edges_data is None:
                        continue

                    multi_edges_values = multi_edges_data.items()

                    # Iterate edges with nodes U and V.
                    for sub_key, sub_data in multi_edges_values:
                        # if for some reason the edge data is empty, continue
                        try:
                            if not sub_data: continue
                        except Exception as ex:
                            continue

                        if sub_data["is_sidewalk"]:

                            sub_m = sub_data["bearing"]
                            if sub_m is None:
                                continue
                            sub_geometry = sub_data["geometry"]

                            # Remove the edge if the sidewalk and the street have a slope <= a threshold
                            abs_angle_diff = abs(m - sub_m)
                            t_half = slope_threshold / 2
                            if (
                                (abs_angle_diff >= (180 - t_half) and abs_angle_diff <= (180 + t_half)) or 
                                (abs_angle_diff >= (360 - t_half)) or 
                                (abs_angle_diff <= t_half)
                            ):
                                
                                # Calculate a buffer of a set meters (street_buffer) and check if the sub_geometry intersects
                                buffer_intersects = utils.intersects(street_buffer, sub_geometry)
                                if buffer_intersects:                                        
                                    # The sidewalk must be at least 20% as large as the street to remove it
                                    if sub_geometry.length > (data["geometry"].length * 0.2):
                                        # Remove the edge from the original graph.
                                        if original_graph.has_edge(u,v,key):
                                            rem_u = u
                                            rem_v = v
                                            rem_key = key
                                        
                                        # print("removing")
                                        # print(rem_u,rem_v)
                                        # print(sub_e_u,sub_e_v)
                                        # print(data,sub_data)
                                        # print()
                                        if assess:
                                            do_assessment = True
                                            remove = False
                                        else:
                                            try:
                                                original_graph.remove_edge(rem_u, rem_v, rem_key)
                                                remove = True
                                            except:
                                                remove = True

                    if remove: break    

        if remove == False:  
            keep_edges.append([u,v,key,do_assessment])

    return keep_edges

def apply_pedestrian_filters(edges_g: geopd.GeoDataFrame):
    # Remove streets with no access
    edges_g = edges_g.loc[~(edges_g["access"].isin(["no", "customers", "private"]))]

    # Keep only roads where the "foot" attribute is pedestrian (designated, yes, permissive) or unknown (None)
    edges_g = edges_g.loc[
        (edges_g["foot"].isin(["designated", "yes", "permissive", None]))
    ]

    # Remove multiple roads that are not pedestrian - "foot" is not specified
    edges_g = edges_g.loc[
        ~((edges_g["highway"].isin(["primary", "secondary", "trunk", "primary_link", "secondary_link", "trunk_link"])) &
        (edges_g["sidewalk"] == "no"))
    ]
    return edges_g
    

def process_pedestrian_graph(gdf: geopd.GeoDataFrame, assessment=False, slope_threshold=None, dist_threshold=None):
    """
    Undirected graph
    """
    if slope_threshold is None:
        slope_threshold = SLOPE_THRESHOLD
    if dist_threshold is None:
        dist_threshold = DIST_THRESHOLD
    # Reset index twice to remove the u,v,k relationship and assign a unique index to
    # each node and edge
    nodes_g = gdf[0].reset_index().reset_index()
    edges_g = gdf[1].reset_index().reset_index()

    nodes_g = nodes_g.replace(np.nan, None)
    edges_g = edges_g.replace(np.nan, None)

    # Create columns if they do not exist
    edge_cols = list(edges_g.columns)
    for field in PEDESTRIAN_FIELDS:
        if field not in edge_cols:
            edges_g[field] = None

    # keep only the necessary columns
    edges_g = edges_g[["u","v","key","geometry"] + PEDESTRIAN_FIELDS]
    edge_cols = list(edges_g.columns)

    # Apply pedestrian filters
    edges_g = apply_pedestrian_filters(edges_g)

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
    remade = ox.simplify_graph(remade, edge_attrs_differ=["foot", "highway", "footway"])
    
    print("finish graph simplify")
    remade = ox.add_edge_bearings(remade)
    # convert to undirected
    remade = remade.to_undirected()
    simplified = copy.deepcopy(remade)

    # get again the remade nodes and edges for the spatial index
    nodes_g, edges_g = ox.graph_to_gdfs(remade)

    foot_types = ["designated", "yes", "permissive"]

    # create a simplified version of the graph only with required information

    simplified_properties = ["highway", "bearing", "geometry", "foot", "footway"]
    for u,v,key,data in simplified.edges(data=True, keys=True):
        # remove all other properties of each edge
        data.clear()
        original_edge = remade[u][v][key]
        if "foot" not in original_edge: original_edge["foot"] = None
        if "footway" not in original_edge: original_edge["footway"] = None

        for prop in simplified_properties:
            simplified[u][v][key][prop] = original_edge[prop]

        # define if the pedestrian segment is a sidewalk
        simplified[u][v][key]["is_sidewalk"] = (
            (original_edge["highway"] in PEDESTRIAN_TYPES) and (original_edge["footway"] in ["sidewalk","crossing"]) or
            (original_edge["highway"] == "cycleway" and original_edge["foot"] in foot_types) or
            (original_edge["highway"] == "footway" and original_edge["footway"] is None)
        )
    
    # Copy to modify edges it without tampering the original
    edges = simplified.edges(data=True, keys=True)

    street_types = ["primary", "secondary", "tertiary", "residential", "unclassified", "primary_link", "secondary_link", "trunk", "trunk_link"]

    # Remove streets that have a separately mapped sidewalks but the street does not specified that
    #   the street has a separate sidewalk.
    # Paralellize the process to speed up computations
    edge_partition = []
    max_partition_size = 30000
    print(f'edges: {len(edges)}')
    partitions = math.ceil(len(edges) / max_partition_size)

    for i in range(0, partitions):
        all_edges = list(edges)
        new_partition = all_edges[i * max_partition_size : (i+1) * max_partition_size]
        edge_partition.append(new_partition)

    print(f"Number of partitions: {len(edge_partition)}")
    parallel = True
    if parallel:
        graph_futures = []
        # Put the simplified graph as a ray object to be passed as reference instead of as value
        simplified_ref = ray.put(simplified)
        _pedestrian_remove_sidewalks_remote = ray.remote(_pedestrian_remove_sidewalks)
        for partition in edge_partition:
            graph_futures.append(_pedestrian_remove_sidewalks_remote.remote(
                street_types, 
                partition, 
                simplified_ref, 
                nodes_g,
                edges_g,
                dist_threshold,
                slope_threshold,
                assess=assessment
            ))

        g_parts = ray.get(graph_futures)
    else:
        g_parts = []
        for partition in edge_partition:
            data_part = _pedestrian_remove_sidewalks(
                street_types, 
                partition, 
                simplified, 
                nodes_g,
                edges_g,
                dist_threshold,
                slope_threshold,
                assess=assessment
            )
            g_parts.append(data_part)


    # merge the parts of the graph into a new undirected graph
    new_g = nx.MultiGraph()
    new_g.graph["crs"] = "epsg:4326"
    for g_part in g_parts:
        for u,v,key,assessment_value in g_part:
            try:
                data = remade[u][v][key]
                # adding speed and travel times to graph
                data["speed_kph"] = 5 # average walking speed of 5km/h

                # convert distance meters to km, and speed km per hour to km per second
                distance_km = data["length"] / 1000
                speed_km_sec = data["speed_kph"] / (60 * 60)
                # calculate edge travel time in seconds
                travel_time = distance_km / speed_km_sec
                data["travel_time"] = travel_time # average walking speed of 5km/h

                if assessment:
                    data["assessment"] = assessment_value

                new_g.add_node(u, **remade._node[u])
                new_g.add_node(v, **remade._node[v])
                new_g.add_edge(u, v, key, **data)

            except:
                print(f"error adding node or edge {u},{v},{key}")

    # remove non-pedestrian ultra-short dead-ends (less than 15 meters)
    print(f"new graph {new_g}")
    remove_singles = []
    for u,v,key,data in new_g.edges(data=True, keys=True):            
        if data["highway"] not in PEDESTRIAN_TYPES:
            if data["length"] < 15: # meters 
                if len(new_g[u]) <= 1 or len(new_g[v]) <= 1: # check if it is a dead-end
                    # print("removing small dead-end")
                    # print(u,v,key)
                    # print()
                    remove_singles.append([u,v,key])
    new_g.remove_edges_from(remove_singles)

    # remove single disconnected edges 
    remove_singles = []
    for u,v,key,data in new_g.edges(data=True, keys=True):
        if len(new_g[u]) <= 1 and len(new_g[v]) <= 1:
            # print("removing single dis edge")
            # print(u,v,key)
            # print()
            remove_singles.append([u,v,key])
    new_g.remove_edges_from(remove_singles)
    
    # remove single disconnected nodes 
    remove_singles = []
    for node_id in new_g.nodes(data=False):
        if len(new_g[node_id]) == 0:
            # print("removing single dis node")
            # print(node_id)
            # print()
            remove_singles.append(node_id)
    new_g.remove_nodes_from(remove_singles)

    ox.add_edge_travel_times

    # return the processed graph
    return new_g