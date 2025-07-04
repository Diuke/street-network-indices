from __future__ import annotations

import scipy
import numpy as np
from pathlib import Path
import pyproj
import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geopd
import scipy.stats
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
    
def steepness(graph: nx.MultiGraph | nx.MultiDiGraph):
    # build the numpy array with the edge values of grade_abs
    abs_grade_np = np.array(list(graph.edges(data="grade_abs", keys=True)), dtype=float)[:,3]

    abs_grade_mean = np.nanmean(abs_grade_np, 0)
    abs_grade_median = np.nanmedian(abs_grade_np, 0)
    abs_grade_range = np.nanmax(abs_grade_np, 0) - np.nanmin(abs_grade_np, 0)
    abs_grade_std = np.nanstd(abs_grade_np, 0)
    abs_grade_iqr = scipy.stats.iqr(abs_grade_np, 0, nan_policy='omit')

    return abs_grade_mean, abs_grade_median, abs_grade_range, abs_grade_std, abs_grade_iqr

def circuity(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False):        
    # transform for conversion from lat/lng to x/y
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    euclidean_sum = 0
    network_sum = 0

    for u,v,key,data in graph.edges(data=True, keys=True):
        p1 = (graph._node[u]["y"], graph._node[u]["x"])
        p2 = (graph._node[v]["y"], graph._node[v]["x"])
        
        euclidean_dist = utils.distance(p1,p2, is_geodetic=True)
        euclidean_sum += euclidean_dist

        if "length" in data:
            s_len = data["length"]
        else:
            s_len += ops.transform(transform, data["geometry"]).length

        network_sum += s_len

        if add_property:
            graph[u][v][key]["circuity"] = s_len / euclidean_dist


    circuity = network_sum / euclidean_sum

    return circuity
    
def street_length(graph: nx.MultiGraph | nx.MultiDiGraph, min_length: float = 10):
    """
    Returns:
    --------
    tuple ():
        Mean, median, range, standard deviation, interquantile range, total
    """
    # transform for conversion from lat/lng to x/y
    transform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    all_lengths = []

    for u,v,key,data in graph.edges(data=True, keys=True):
        if "length" in data:
            s_len = data["length"]
        else:
            s_len = ops.transform(transform, data["geometry"]).length
            graph[u][v][key]["length"] = s_len

        # only consider streets longer than a certain threshold
        if s_len > min_length:
            all_lengths.append(s_len)
        else:
            all_lengths.append(np.nan)

    np_street_len = np.array(all_lengths)
    st_len_mean = np.nanmean(np_street_len, 0)
    st_len_median = np.nanmedian(np_street_len, 0)
    st_len_range = np.nanmax(np_street_len) - np.nanmin(np_street_len)
    st_len_std = np.nanstd(np_street_len, 0)
    st_len_iqr = scipy.stats.iqr(np_street_len, 0, nan_policy='omit')
    st_len_total = np.nansum(np_street_len, 0)
    # get the average with only the amount of valid streets

    return st_len_mean, st_len_median, st_len_range, st_len_std, st_len_iqr, st_len_total


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

def average_walking_road_score(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False):
    road_scores_count = 0
    walk_number_of_edges = graph.number_of_edges()

    for u,v,key,data in graph.edges(data=True, keys=True):
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
        
        # add property
        if add_property:
            graph[u][v][key]["walk_score"] = road_score


    mean_road_score = road_scores_count / walk_number_of_edges
    return mean_road_score

def average_cycling_road_score(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False):
    road_scores_count = 0
    bike_number_of_edges = graph.number_of_edges()

    for u,v,key,data in graph.edges(data=True,keys=True):
        road_score = 0
        road_type = data["highway"]
        try:
            max_speed = int(data["max_speed"])
        except: max_speed = None
        if "cycleway" in data:
            cycleway = data["cycleway"]
        else: 
            cycleway = None
        inclination = data["grade_abs"]
        is_cycleway = road_type == "cycleway"

        if is_cycleway: 
            road_score = 5
        elif cycleway is not None:
            if cycleway != "no": #includes lane
                road_score = 4
        elif "trunk" in road_type:
            road_score = 0
        elif "primary" in road_type:
            road_score = 1
        elif "secondary" in road_type:
            road_score = 2
        elif "tertiary" in road_type:
            road_score = 3
        elif road_type == "residential" or road_type == "living_street":
            road_score = 4
        else: 
            road_score = 3

        # decrease score depending on road speed
        if max_speed is not None:
            if max_speed >= 50:
                road_score -= 1
            elif max_speed >= 80:
                road_score -= 2
            
            if road_score < 0: road_score = 0

        # decrease score depending on inclination
        if inclination is not None:
            inclination_percentage = inclination * 100
            if inclination_percentage < 4: #easy inclination
                road_score -= 0
            elif inclination_percentage >= 4 and inclination_percentage < 7: # moderate inclination
                road_score -= 1
            elif inclination_percentage >= 7 and inclination_percentage < 9: # challenging
                road_score -= 2
            elif inclination_percentage > 9: # hard
                road_score -= 3
        
        if road_score < 0: road_score = 0

        # sum of road scores
        road_scores_count += road_score
        
        # add property
        if add_property:
            graph[u][v][key]["bike_score"] = road_score


    mean_road_score = road_scores_count / bike_number_of_edges
    return mean_road_score

def link_node_ratio(graph: nx.MultiGraph | nx.MultiDiGraph):
    if graph.is_directed():
        g = graph.to_undirected()
    else:
        g = graph.copy()

    links = g.number_of_edges()
    nodes = g.number_of_nodes()
    return links / nodes


def intersection_density(graph: nx.MultiGraph | nx.MultiDiGraph, area: float):
    """
    Returns
    -------
    tuple: (intersection density, number of intersections)
    """
    intersections = 0
    if graph.is_directed():
        # calculate intersections for directed graph
        #  traverse entire graph
        for node_id in graph.nodes():
            incident = graph.in_edges(node_id)
            adjacent = graph.out_edges(node_id)

            distinct_edges = set()
            for inc in incident: distinct_edges.add(inc)
            for adj in adjacent: distinct_edges.add(adj)

            # intersections or dead-ends
            if len(distinct_edges) >= 3 or len(distinct_edges) == 1:
                intersections += 1

    else:
        # calculate intersections for undirected graph
        for node_id in graph.nodes():
            degree = graph.degree(node_id)
            
            # intersections or dead-ends
            if degree >= 3 or degree == 1:
                intersections += 1
    
    return (intersections / area), intersections


def connectivity(graph: nx.MultiGraph | nx.MultiDiGraph):
        if graph.is_directed():
            # calculate connectivity for directed graph
            natural_streets_walking_graph = generalize.named_streets_generalization(graph, is_directed=True)

        else:
            # calculate connectivity for undirected graph
            natural_streets_walking_graph = generalize.named_streets_generalization(graph, is_directed=False)

        connectivity = []
        for node_id, data in natural_streets_walking_graph.nodes(data=True):
            edges_len = natural_streets_walking_graph.degree(node_id)
            connectivity.append(edges_len)

        connectivity_array = np.array(connectivity)
        connectivity_mean = np.mean(connectivity_array)
        connectivity_std = np.std(connectivity_array)
        # connectivity_range = connectivity_array.max() - connectivity_array.min()
        # connectivity_90 = (connectivity_range / 10) * 9
        # connectivity_top_10_len = len(list(filter(lambda x: x > connectivity_90, connectivity_array)))

        return connectivity_mean, connectivity_std

