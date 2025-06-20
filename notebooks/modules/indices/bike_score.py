# Bike score takes into account: 
# bike lanes (infrastructure), 
# hills (slope), 
# connectivity (distance to POIs).
from __future__ import annotations

import networkx as nx
import osmnx as ox

def path_score(edge_data):
    # infrastructure
    # We weight bike paths 2X as valuable as bike lanes and 3X as valuable as shared infrastructure. 
    # This creates a raw value that we normalize to a score between 0 - 100 based on an average of 
    # the highest Bike Lane Scores that we sampled.
    if edge_data["highway"] == "cycleway" or edge_data["highway"] == "path":
        return 100 # value of 100
    
    elif edge_data["cycleway"] == "lane":
        return 25.37313432835821 # value of 50
    
    else:
        return 0 # value of 33

def slope_score(edge_data):
    #A grade of 10% - 2% is given a score of 0 - 100.
    slope_perc = edge_data["grade_abs"] * 100
    min_slope = 2
    max_slope = 10

    if slope_perc <= min_slope:
        return 0
    if slope_perc >= max_slope:
        return 100
    
    return ((slope_perc - min_slope) * 100) / (max_slope - min_slope)

def average_cycling_road_score(graph: nx.MultiGraph | nx.MultiDiGraph, add_property=False):
    road_scores_count = 0
    bike_number_of_edges = graph.number_of_edges()

    for u,v,key,data in graph.edges(data=True,keys=True):
        # check infrastructure
        infra_score = path_score(data)
        # check slope
        slope = slope_score(data)
        # calculate
        score = (infra_score + slope) / 2

        road_scores_count += score

        # add property
        if add_property:
            graph[u][v][key]["bike_score"] = score

    mean_road_score = road_scores_count / bike_number_of_edges
    return mean_road_score



        
    