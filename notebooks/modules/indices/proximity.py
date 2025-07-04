
from __future__ import annotations

import scipy
import math
import ray
import scipy.stats
import shapely
import numpy as np
import networkx as nx
import osmnx as ox
from queue import PriorityQueue
from modules.download import poi

from modules import utils

def graph_walk_distance(g: nx.MultiDiGraph | nx.MultiGraph, initial_node, distance:int=None, attribute=None, add_property=True, attribute_type:str="categorical"):
    """
    BFS graph walk.
    Distance in meters.
    
    attribute_type can be "num" or "categorical"
    """
    visited_edges = set() #u,v,key
    visited_nodes = set()
    attribute_set = set()
    attribute_count = 0
    
    visit_queue = PriorityQueue()
    # use unique index as second criteria so stations at the same distance dont break the queue
    unique_index = 0
    visit_queue.put((0.0, unique_index, {"node_id": initial_node, "distance": 0.0, "parent": None, "key": None}))
    
    while not visit_queue.empty():
        top = visit_queue.get()
        current  = top[2]
        current_id = current["node_id"]
        current_distance = current["distance"]
        current_parent = current["parent"]
        current_edge_key = current["key"]

        if current_parent is not None and current_edge_key is not None:
            edge_hash = f"{current_parent},{current_id}"
            if edge_hash not in visited_edges:
                visited_edges.add(edge_hash)
                if not g.is_directed():
                    visited_edges.add(f"{current_id},{current_parent}")
 
                
                if attribute is not None:
                    attribute_value = g[current_parent][current_id][current_edge_key][attribute]
                    if attribute_value is not None:
                        if attribute_type == "categorical":
                            attribute_set = attribute_set.union(attribute_value)
                        elif attribute_type == "num":
                            attribute_count += attribute_value

        if current_id in visited_nodes:
            continue
        else:
            visited_nodes.add(current_id)

        # this ensures that partial streets are added to the 15-minute walk
        if current_distance >= distance:
            continue 

        incident = g.adj[current_id]
        for adj_node in incident:
            adj_node_keys = g.get_edge_data(current_id, adj_node).items()
            # determine shorter u,v,key
            short_key = None
            short_distance = 9999999
            for key, edge_data in adj_node_keys:
                edge_distance = float(edge_data["length"])
                if edge_distance < short_distance:
                    short_key = key
                    short_distance = edge_distance

            new_distance = current_distance + short_distance # meters
            # ordered queue, so the closes element is visited first
            unique_index += 1
            visit_queue.put((new_distance, unique_index, {"node_id": adj_node, "distance": new_distance, "parent": current_id, "key": short_key}))

    if add_property:
        return_value = None
        if attribute_type == "categorical":
            g._node[initial_node][f"{attribute}_count"] = len(attribute_set)
            return_value = attribute_set
        elif attribute_type == "num":
            g._node[initial_node][f"{attribute}_count"] = attribute_count
            return_value = attribute_count

    if attribute is not None: 
        return visited_edges, return_value
    else: 
        return visited_edges

def graph_walk_time(g: nx.MultiDiGraph | nx.MultiGraph, initial_node, time=None, attribute=None, add_property=True, attribute_type:str="categorical"):
    """
    BFS graph walk.
    Time in minutes

    attribute_type can be "num" or "categorical"
    """

    time_seconds = time * 60

    visited_edges = set() #u,v,key
    visited_nodes = set()
    attribute_set = set()
    attribute_count = 0
    
    visit_queue = PriorityQueue()
    # use unique index as second criteria so stations at the same distance dont break the queue
    unique_index = 0
    visit_queue.put((0.0, unique_index, {"node_id": initial_node, "time": 0.0, "parent": None, "key": None}))
    
    while not visit_queue.empty():
        top = visit_queue.get()
        current  = top[2]
        current_id = current["node_id"]
        current_time = current["time"]
        current_parent = current["parent"]
        current_edge_key = current["key"]

        if current_parent is not None and current_edge_key is not None:
            edge_hash = f"{current_parent},{current_id}"
            if edge_hash not in visited_edges:
                visited_edges.add(edge_hash)
                if not g.is_directed():
                    visited_edges.add(f"{current_id},{current_parent}")
 
                
                if attribute is not None:
                    attribute_value = g[current_parent][current_id][current_edge_key][attribute]
                    if attribute_value is not None:
                        if attribute_type == "categorical":
                            attribute_set = attribute_set.union(attribute_value)
                        elif attribute_type == "num":
                            attribute_count += attribute_value

        if current_id in visited_nodes:
            continue
        else:
            visited_nodes.add(current_id)

        # this ensures that partial streets are added to the 15-minute walk
        if current_time >= time_seconds:
            continue 

        incident = g.adj[current_id]
        for adj_node in incident:
            adj_node_keys = g.get_edge_data(current_id, adj_node).items()
            # determine shorter u,v,key
            short_key = None
            short_time = 9999999
            for key, edge_data in adj_node_keys:
                edge_time = float(edge_data["travel_time"])
                if edge_time < short_time:
                    short_key = key
                    short_time = edge_time
            
            new_time = float(current_time + short_time) # meters
            # ordered queue, so the closes element is visited first
            unique_index += 1
            visit_queue.put((new_time, unique_index, {"node_id": adj_node, "time": new_time, "parent": current_id, "key": short_key}))
                    
    if add_property:
        return_value = None
        if attribute_type == "categorical":
            g._node[initial_node][f"{attribute}_count"] = len(attribute_set)
            return_value = attribute_set
        elif attribute_type == "num":
            g._node[initial_node][f"{attribute}_count"] = attribute_count
            return_value = attribute_count

    if attribute is not None: 
        return visited_edges, return_value
    else: 
        return visited_edges
    

def __process_nodes_time(node_list, graph, walk_time:int=15, attribute="amenities", attribute_type="categorical"):
    """
    attribute type categorical or num
    """
    print(f'processing {len(node_list)} nodes')
    #total = 0
    attribute_count_list = []
    for node in node_list:
        initial = node
        visited_edges, am_count = graph_walk_time(graph, initial, walk_time, attribute=attribute)
        attribute_count_list.append(am_count)
        #total += len(am_count)
    return attribute_count_list

def __process_nodes_distance(node_list, graph, walk_distance:int=400, attribute="amenities", attribute_type="categorical"):
    print(f'processing {len(node_list)} nodes')
    attribute_count_list = []
    for node in node_list:
        initial = node
        visited_edges, am_count = graph_walk_distance(graph, initial, walk_distance, attribute=attribute, attribute_type=attribute_type)
        attribute_count_list.append(am_count)
    return attribute_count_list

def __process_visited_edges(edge: str):
    edge_str = edge.split(",")
    return ( int(edge_str[0]), int(edge_str[1]), int(edge_str[2]) )


def proximity_to_pois(g: nx.MultiDiGraph | nx.MultiGraph, geom, walk_time:int=15, parallelize:bool=True, max_partition_size:int=10000):
    pois = poi.download_pois(geom)
    pois.reset_index(inplace=True)
    poi.set_pois_to_edges(g, pois)

    all_nodes = g.nodes()

    if parallelize:
        node_partition = []
        print(f'nodes: {len(all_nodes)}')
        partitions = math.ceil(len(all_nodes) / max_partition_size)

        node_list = list(all_nodes)
        for i in range(0, partitions):
            new_partition = node_list[i * max_partition_size : (i+1) * max_partition_size]
            node_partition.append(new_partition)

        original_graph = ray.put(g)
        node_number = g.number_of_nodes()

        remote_process_nodes = ray.remote(__process_nodes_time)
        graph_futures = []


        for partition in node_partition:
            graph_futures.append(remote_process_nodes.remote(
                partition, original_graph, walk_time=walk_time
            ))

        g_parts = ray.get(graph_futures)

    else: 
        g_parts = [
            __process_nodes_time(all_nodes, g, walk_time=walk_time)
        ]

    pois_per_node = []
    for poi_list in g_parts:
        for node_pois in poi_list:
            pois_per_node.append(len(node_pois))

    np_pois = np.array(pois_per_node)
    np_pois_mean = np.nanmean(np_pois, 0)
    np_pois_median = np.nanmedian(np_pois, 0)
    np_pois_range = np.nanmax(np_pois, 0) - np.nanmin(np_pois, 0)
    np_pois_std = np.nanstd(np_pois, 0)
    np_pois_iqr = scipy.stats.iqr(np_pois, 0, nan_policy='omit')
    # get the average with only the amount of valid streets

    return np_pois_mean, np_pois_median, np_pois_range, np_pois_std, np_pois_iqr

def proximity_to_pois_distance(g: nx.MultiDiGraph | nx.MultiGraph, walk_distance:int=400, parallelize:bool=True, max_partition_size:int=10000):
    all_nodes = g.nodes()

    if parallelize:
        node_partition = []
        print(f'nodes: {len(all_nodes)}')
        partitions = math.ceil(len(all_nodes) / max_partition_size)

        node_list = list(all_nodes)
        for i in range(0, partitions):
            new_partition = node_list[i * max_partition_size : (i+1) * max_partition_size]
            node_partition.append(new_partition)

        original_graph = ray.put(g)
        node_number = g.number_of_nodes()

        remote_process_nodes = ray.remote(__process_nodes_distance)
        graph_futures = []


        for partition in node_partition:
            graph_futures.append(remote_process_nodes.remote(
                partition, original_graph, walk_distance=walk_distance
            ))

        g_parts = ray.get(graph_futures)

    else: 
        g_parts = [
            __process_nodes_distance(all_nodes, g, walk_distance=walk_distance)
        ]

    total_pois = 0
    for n in g_parts:
        total_pois += n

    mean_pois = total_pois / node_number 
    return mean_pois

def has_station_in_distance(graph, initial_node, max_distance):
    
    visited_nodes = set()
    visited_edges = set() 
    
    visit_queue = PriorityQueue()
    # use unique index as second criteria so stations at the same distance dont break the queue
    unique_index = 0
    visit_queue.put((0, unique_index, {"node_id": initial_node, "distance": 0, "parent": None, "key": None}))
    
    while not visit_queue.empty():
        current = visit_queue.get()[2]
        current_id = current["node_id"]
        current_distance = current["distance"]
        current_parent = current["parent"]
        current_edge_key = current["key"]

        if current_parent is not None:
            edge_hash = f"{current_parent},{current_id}"
            if edge_hash not in visited_edges:
                visited_edges.add(edge_hash)
                if not graph.is_directed():
                    # If is undirected, add also the inverse edge so it is not visited
                    visited_edges.add(f"{current_id},{current_parent}")
                
                station_count = graph[current_parent][current_id][current_edge_key]["public_transport_count"]
                if station_count > 0:
                    graph._node[initial_node]["public_transport_prox"] = True
                    return True

        if current_id in visited_nodes:
            continue
        else:
            visited_nodes.add(current_id)

        # this ensures that partial streets are added to the 15-minute walk
        if current_distance >= max_distance:
            continue 

        incident = graph.adj[current_id]
        for adj_node in incident:
            adj_node_keys = graph.get_edge_data(current_id, adj_node).items()
            # determine shorter u,v,key
            short_key = None
            short_distance = 9999999
            for key, edge_data in adj_node_keys:
                edge_distance = float(edge_data["length"])
                if edge_distance < short_distance:
                    short_key = key
                    short_distance = edge_distance

            new_distance = current_distance + short_distance # meters

            if new_distance < max_distance:
                # ordered queue, so the closes element is visited first
                unique_index += 1
                visit_queue.put((new_distance, unique_index, {"node_id": adj_node, "distance": new_distance, "parent": current_id, "key": short_key}))

    return False # if it does not find a public transport edge
                

def proximity_to_public_transport(graph:nx.MultiDiGraph | nx.MultiGraph, public_transport_g:nx.MultiDiGraph, distance:int, parallelize=True, max_partition_size=10000):
    public_nodes = ox.graph_to_gdfs(public_transport_g, nodes=True, edges=False)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    number_of_nodes = graph.number_of_nodes()
    
    nx.set_edge_attributes(graph, 0, "public_transport_count")
    buffer_distance = 100 #meters
    buffer_meters = utils.m_to_deg(buffer_distance)

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges = edges.reset_index()

    all_count = []

    for idx, poi in public_nodes.iterrows():
        poi_geom = poi["geometry"]

        # assign the amenity to the closest edge
        closer = edges.sindex.query(poi_geom, "dwithin", distance=buffer_meters, sort=True)
        if len(closer) > 0:
            min_distance = 9999
            min_close = None

            for close in closer:
                close_edge_data = edges.iloc[close]
                geom_distance = shapely.distance(close_edge_data["geometry"], poi["geometry"])
                if geom_distance < min_distance:
                    min_distance = geom_distance
                    min_close = close
                
            closer_edge = edges.iloc[min_close]

            u = closer_edge["u"]
            v = closer_edge["v"]
            key = closer_edge["key"]
            graph[u][v][key]["public_transport_count"] += 1



    all_nodes = graph.nodes()

    if parallelize:
        node_partition = []
        print(f'nodes: {len(all_nodes)}')
        partitions = math.ceil(len(all_nodes) / max_partition_size)

        node_list = list(all_nodes)
        for i in range(0, partitions):
            new_partition = node_list[i * max_partition_size : (i+1) * max_partition_size]
            node_partition.append(new_partition)

        original_graph = ray.put(graph)
        node_number = graph.number_of_nodes()

        remote_process_nodes = ray.remote(__process_nodes_distance)
        graph_futures = []

        for partition in node_partition:
            graph_futures.append(remote_process_nodes.remote(
                partition, original_graph, walk_distance=distance, attribute="public_transport_count", attribute_type="num"
            ))

        g_parts = ray.get(graph_futures)

    else: 
        g_parts = [
            __process_nodes_distance(all_nodes, graph, walk_distance=distance, attribute="public_transport_count", attribute_type="num")
        ]

    pois_per_node = []
    for poi_list in g_parts:
        for node_pois in poi_list:
            pois_per_node.append(node_pois)

    np_pois = np.array(pois_per_node)
    np_pois_mean = np.nanmean(np_pois, 0)
    np_pois_median = np.nanmedian(np_pois, 0)
    np_pois_range = np.nanmax(np_pois, 0) - np.nanmin(np_pois, 0)
    np_pois_std = np.nanstd(np_pois, 0)
    np_pois_iqr = scipy.stats.iqr(np_pois, 0, nan_policy='omit')

    return np_pois_mean, np_pois_median, np_pois_range, np_pois_std, np_pois_iqr
