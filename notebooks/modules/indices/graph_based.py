import scipy
import numpy as np
from pathlib import Path
import pyproj
import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geopd
import shapely
from geopy import distance as geopy_distance
from shapely import ops, LineString
import requests
from itertools import chain
from modules import utils
from networkx import bfs_edges
from modules import generalize
import sys


def calculate_all_indices(graph: nx.MultiGraph | nx.MultiDiGraph, area: float):
    """
    Calculate all available indices in an optimal, single-graph traversal.
    """
    # transform for conversion from lat/lng to x/y
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    indices = [
        "circuity", "orientation_entropy", "average_steepness"
    ]
    # store the results into a dictionary.
    result = {}
    for ind in indices:
        # create the index result for each index 
        result[ind] = None


    edges_len = graph.number_of_edges()
    nodes_len = graph.number_of_nodes()

    euclidean_sum = 0
    network_sum = 0

    for u,v,key,data in graph.edges(data=True, keys=True):
        # calculations for circuity
        p1 = (graph._node[u]["geometry"].coords[0][1], graph._node[u]["geometry"].coords[0][0])
        p2 = (graph._node[v]["geometry"].coords[0][1], graph._node[v]["geometry"].coords[0][0])

        euclidean_dist = utils.distance(p1,p2, is_geodetic=True)
        euclidean_sum += euclidean_dist
            
        # calculations for steepness
        elevation_sum += data["grade_abs"]

        # calculations for network total length
        if "length" in data:
            network_sum += data["length"]
        else:
            network_sum += ops.transform(transform, data["geometry"]).length



    result["circuity"] = network_sum / euclidean_sum
    result["average_steepness"] = elevation_sum / edges_len
    result["orientation_entropy"] = ox.bearing.orientation_entropy(graph, min_length=15, weight="length")
    result["road_density"] = network_sum / area
    result["average_street_lenght"] = network_sum / edges_len

    return result

def orientation_entropy(graph: nx.MultiGraph | nx.MultiDiGraph):
    orientation_entropy = ox.bearing.orientation_entropy(graph, min_length=15, weight="length")
    return orientation_entropy
    
def average_steepness(graph: nx.MultiGraph | nx.MultiDiGraph):
    elevation_sum = 0
    edges_len = graph.number_of_edges()
    for u,v,key,data in graph.edges(data=True, keys=True):
        elevation_sum += data["grade_abs"]

    average_steepness = elevation_sum / edges_len
    return average_steepness

def circuity(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False):        
    # transform for conversion from lat/lng to x/y
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    euclidean_sum = 0
    network_sum = 0

    for u,v,key,data in graph.edges(data=True, keys=True):
        p1 = (graph._node[u]["geometry"].coords[0][1], graph._node[u]["geometry"].coords[0][0])
        p2 = (graph._node[v]["geometry"].coords[0][1], graph._node[v]["geometry"].coords[0][0])
        
        euclidean_dist = utils.distance(p1,p2, is_geodetic=True)
        euclidean_sum += euclidean_dist

        if "length" in data:
            s_len = data["length"]
        else:
            s_len += ops.transform(transform, data["geometry"]).length

        network_sum += s_len

        if add_property:
            graph[u][v][key]["circuity"] = network_sum / euclidean_sum


    circuity = network_sum / euclidean_sum

    return circuity
    
def average_street_length(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False, min_length: float = 10):
    # transform for conversion from lat/lng to x/y
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    network_sum = 0
    valid_streets = 0

    for u,v,key,data in graph.edges(data=True, keys=True):
        if "length" in data:
            s_len = data["length"]
        else:
            s_len += ops.transform(transform, data["geometry"]).length

        # only consider streets longer than a certain threshold
        if s_len > min_length:
            network_sum += s_len
            valid_streets += 1

    # get the average with only the amount of valid streets
    average_street_length = network_sum / valid_streets

    return average_street_length










def data_has_sidewalk(data):
    if "sidewalk" in data:
        if data["sidewalk"] is not None:
            if data["sidewalk"] == "both" or data["sidewalk"] == "separate" or data["sidewalk"] == "left" or data["sidewalk"] == "right" or data["sidewalk"] == "yes":
                return True
            
    if "sidewalk:both" in data:
        if data["sidewalk:both"] is not None:
            if data["sidewalk:both"] == "separate" or data["sidewalk:both"] == "yes":
                return True

    if "sidewalk:left" in data:
        if data["sidewalk:left"] is not None:
            if data["sidewalk:left"] == "separate" or data["sidewalk:left"] == "yes":
                return True
        
    if "sidewalk:right" in data:
        if data["sidewalk:right"] is not None:
            if data["sidewalk:right"] == "separate" or data["sidewalk:right"] == "yes":
                return True
        
    return False


def calculate_walking_metrics(walk_g: nx.MultiGraph | nx.MultiDiGraph, drive_g: nx.MultiGraph | nx.MultiDiGraph, area: float):
    # generalizer = generalize.Generalize()
    # natural_streets_walking_graph = generalizer.named_streets_generalization(walk_g)
    
    walk_number_of_edges = walk_g.number_of_edges()
    # drive_number_of_edges = drive_g.number_of_edges()
    # walk_drive_ratio = walk_number_of_edges / drive_number_of_edges

    # connectivity = []
    # for node_id, data in natural_streets_walking_graph.nodes(data=True):
    #     edges_len = len(natural_streets_walking_graph[node_id])
    #     connectivity.append(edges_len)

    # connectivity_array = np.array(connectivity)
    # connectivity_mean = np.mean(connectivity_array)
    # connectivity_std = np.std(connectivity_array)
    # connectivity_range = connectivity_array.max() - connectivity_array.min()
    # connectivity_90 = (connectivity_range / 10) * 9
    # connectivity_top_10_len = len(list(filter(lambda x: x > connectivity_90, connectivity_array)))

    # intersection count for undirected graph
    intersections = 0
    for node_id in walk_g.nodes():
        if len(walk_g[node_id]) >= 3:
            intersections += 1

    road_scores_count = 0
    for u,v,data in walk_g.edges(data=True):
        road_score = 0

        road_type = data["highway"]
        try:
            max_speed = int(data["max_speed"])
        except: max_speed = None
        has_sidewalk = data_has_sidewalk(data)
        is_sidewalk = (data["footway"] == "sidewalk") or (road_type == "footway")

        if is_sidewalk or road_type == "path": 
            road_score = 5
        elif "trunk" in road_type and not has_sidewalk:
            road_score = 0
        elif "primary" in road_type and not has_sidewalk:
            road_score = 1
        elif "secondary" in road_type and not has_sidewalk:
            road_score = 2
        elif "tertiary" in road_type:
            road_score = 3
        elif road_type == "residential" or road_type == "living_street":
            road_score = 4
        else: 
            road_score = 3
        
        # Improve rating if it has a sidewalk (not mapped separately)
        if has_sidewalk:
            road_score = (road_score + 2) % 5

        # decrease score depending on road speed
        if max_speed is not None:
            if max_speed >= 30:
                road_score -= 0.5
            elif max_speed >= 50:
                road_score -= 1
            elif max_speed >= 80:
                road_score -= 2
            
            if road_score < 0: road_score = 0

        # sum of road scores
        road_scores_count += road_score


    mean_road_score = road_scores_count / walk_number_of_edges
    # intersection_density = intersections / area
    return {
        # "walk_connectivity_mean": connectivity_mean,
        # "walk_connectivity_std": connectivity_std,
        # "walk_connectivity_top_10p": connectivity_top_10_len
        "walk_connectivity_mean": 0,
        "walk_connectivity_std": 0,
        "walk_connectivity_top_10p": 0,

        "mean_road_score": mean_road_score,
        # "walk_connectivity_mean": connectivity_mean,
        # "walk_connectivity_std": connectivity_std,
        # "walk_connectivity_top_10p": connectivity_top_10_len
        "walk_connectivity_mean": 0,
        "walk_connectivity_std": 0,
        "walk_connectivity_top_10p": 0
    }