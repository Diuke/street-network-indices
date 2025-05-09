import copy
import math
import ray
import numpy as np
import networkx as nx
import geopandas as geopd
import osmnx as ox
from modules import utils
from modules.modes import pedestrian

def _mark_potential_streets(
        # street_types:list[str], 
        process_edges:list[dict["u":any,"v":any,"key":any,"data":any]], # for paralellism

        original_graph:nx.MultiGraph, 
        nodes_gdf:geopd.GeoDataFrame, # for the spatial index
        edges_gdf:geopd.GeoDataFrame, # for the spatial index

        dist_threshold:int,
        slope_threshold:int

    ):
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
    # approximately convert to degrees -> 1 deg = 111595.75 meters
    # sum 5 meters for the erosion
    DEG_CONVERT = 111595.75 
    buffer_distance = dist_threshold / DEG_CONVERT
    # 5 meters erosion
    erode_distance = 5 / DEG_CONVERT

    street_types = ["primary", "secondary", "tertiary", "residential", "unclassified", "primary_link", "secondary_link", "trunk", "trunk_link", "service"]
    pedestrian_types = []

    # nx.set_edge_attributes(original_graph, degree, "degree")

    print(f"starting with {len(process_edges)} edges")
    for u,v,key,data in process_edges:        
        # if for some reason the data of the edge is empty, continue
        
        try:
            if not data: continue
        except Exception as ex:
            print(f"error - skip {u},{v}")

        # check if the type of street is one of the types that are markable


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
            # if it is a self loop (bearing undefined) - Self loops pose no problem
            m = data["bearing"]
            if m is None:
                original_graph[u][v][key]["assess"] = False
                original_graph[u][v][key]["note"] = ""
                continue
            
            #if the street is too small and the erosion is larger than the street continue - keep and skip
            if data["geometry"].length  <= (erode_distance * 2):
                original_graph[u][v][key]["assess"] = False
                original_graph[u][v][key]["note"] = ""
                continue

            # meters to degree approximate conversion
            street_buffer = utils.buffer(data["geometry"], buffer_distance, erode_distance)

            # Create a subgraph with the edges that intersect the buffer
            # It uses geopandas spatial index for speedup
            nearest_idx = edges_gdf.sindex.query(street_buffer, predicate="intersects")
            edges_in_buffered_graph = edges_gdf.iloc[nearest_idx]
            subgraph = original_graph.edge_subgraph(list(edges_in_buffered_graph.index))

            # Iterate the edges of the subgraph
            for sub_e_u, sub_e_v in subgraph.edges():
                # The nodes are integers, but come as strings
                sub_e_u = int(sub_e_u)
                sub_e_v = int(sub_e_v)

                # Check that the edge is not the same edge that is being tested
                if u == sub_e_u and v == sub_e_v: continue
                if v == sub_e_u and u == sub_e_v: continue
                if v == sub_e_u and u == sub_e_v: continue

                # Get the edge data of the edge(s) to test (multiple in case of multiedge)
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

                        # Mark the edge if the sidewalk and the street have a slope <= a threshold
                        abs_angle_diff = abs(m - sub_m)
                        t_half = slope_threshold / 2
                        if (
                            (abs_angle_diff >= (180 - t_half) and abs_angle_diff <= (180 + t_half)) or 
                            (abs_angle_diff >= (360 - t_half)) or 
                            (abs_angle_diff <= t_half)
                        ):
                            
                            # The sidewalk must be at least 25% as large as the street to remove it
                            if sub_geometry.length > (data["geometry"].length * 0.2):
                                # Mark the edge on the original graph.
                                
                                print("assess street")
                                original_graph[u][v][key]["assess"] = True
                                original_graph[u][v][key]["note"] = "Nearby sidewalk"


def assess_pedestrian_graph(gdf: geopd.GeoDataFrame, simplify=False):
    """
    Do some processes for assessment of pedestrian graphs

    1. Check for streets with non-specified separated sidewalk with nearby potential sidewalks.
    
    2. Check disconnected pedestrian sidewalks.
    """
    
    slope_threshold = pedestrian.SLOPE_THRESHOLD
    dist_threshold = pedestrian.DIST_THRESHOLD
        
    # Reset index twice to remove the u,v,k relationship and assign a unique index to
    # each node and edge
    nodes_g = gdf[0].reset_index().reset_index()
    edges_g = gdf[1].reset_index().reset_index()

    nodes_g = nodes_g.replace(np.nan, None)
    edges_g = edges_g.replace(np.nan, None)

    # Create columns if they do not exist
    edge_cols = list(edges_g.columns)
    for field in pedestrian.PEDESTRIAN_FIELDS:
        if field not in edge_cols:
            edges_g[field] = None

    # keep only the necessary columns
    edges_g = edges_g[["u","v","key","geometry"] + pedestrian.PEDESTRIAN_FIELDS]
    edge_cols = list(edges_g.columns)
    
    # Apply pedestrian filters
    edges_g = pedestrian.apply_pedestrian_filters(edges_g)

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
    
    # try the simplified
    if simplify:
        # remove interstitial nodes only by highway type
        remade = ox.simplify_graph(remade, edge_attrs_differ=["highway"])

    print("finish graph simplify")
    # bearings are necessary
    remade = ox.add_edge_bearings(remade)
    
    # convert to undirected
    remade = remade.to_undirected()

    # copy for the simple versions that gets sent to the _mark function
    simplified = copy.deepcopy(remade)

    # get again the remade nodes and edges for the spatial index
    nodes_g, edges_g = ox.graph_to_gdfs(remade)

    # create a simplified version of the graph only with required information
    foot_types = ["designated", "yes", "permissive"]
    simplified_properties = ["highway", "bearing", "geometry", "foot", "footway"]
    for u,v,key,data in simplified.edges(data=True, keys=True):
        # remove all other properties of each edge
        data.clear()
        original_edge = remade[u][v][key]
        if "foot" not in original_edge: original_edge["foot"] = None
        if "footway" not in original_edge: original_edge["footway"] = None

        for prop in simplified_properties:
            if prop not in original_edge: original_edge[prop] = None
            simplified[u][v][key][prop] = original_edge[prop]

        # define if the pedestrian segment is a sidewalk
        simplified[u][v][key]["is_sidewalk"] = (
            (original_edge["highway"] in pedestrian.PEDESTRIAN_TYPES) and (original_edge["footway"] in ["sidewalk","crossing"]) or
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
    parallel = False
    if parallel:
        graph_futures = []
        # Put the simplified graph as a ray object to be passed as reference instead of as value
        
        simplified_ref = ray.put(simplified)
        #simplified_ref = ray.put(simplified)
        _pedestrian_remove_sidewalks_remote = ray.remote(_mark_potential_streets)
        for partition in edge_partition:

            graph_futures.append(_pedestrian_remove_sidewalks_remote.remote(
                partition, 
                simplified_ref, 
                nodes_g,
                edges_g,
                pedestrian.DIST_THRESHOLD,
                pedestrian.SLOPE_THRESHOLD,
            ))

        g_parts = ray.get(graph_futures)

    else:
        g_parts = []
        for partition in edge_partition:
            data_part = _mark_potential_streets(
                partition, 
                simplified, 
                nodes_g,
                edges_g,
                pedestrian.DIST_THRESHOLD,
                pedestrian.SLOPE_THRESHOLD,
            )
            g_parts.append(data_part)

    print("Finish processing")
    # add the assessment property to the original graph
    for u,v,key,data in simplified.edges(keys=True, data=True):
        remade[u][v][key]["assess"] = simplified[u][v][key]["assess"] if "assess" in simplified[u][v][key] else False
        remade[u][v][key]["note"] = simplified[u][v][key]["note"] if "note" in simplified[u][v][key] else ""

    # remove single disconnected nodes 
    remove_singles = []
    for node_id in remade.nodes(data=False):
        if len(remade[node_id]) == 0:
            remove_singles.append(node_id)
    remade.remove_nodes_from(remove_singles)

    # return the processed graph
    return remade

def assess_cycling_graph():
    """
    Do some processes for assessment of pedestrian graphs

    1. Check for streets with non-specified separated cycleways with nearby potential cycleways.
    
    2. Check disconnected cycling paths.
    """
    pass

