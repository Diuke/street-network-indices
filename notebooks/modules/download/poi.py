from __future__ import annotations

import shapely
import osmnx as ox
import networkx as nx
import pandas as pd
from modules import utils

POI_FILTER = {
    'amenity': [
        "bar", "biergarten", "cafe", "fast_food", "food_court", "ice_cream", "pub", "restaurant",
        "college", "driving_school", "kindergarten", "language_school", "library", "research_institute", "school", "music_school", "university",
        "fuel", "charging_station", "motorcycle_parking", "bicycle_parking", "parking", "taxi",
        "atm", "bank", "bureau_de_change", "money_transfer",
        "clinic", "dentist", "doctors", "hospital", "nursing_home", "pharmacy", "social_facility", "veterinary",
        "arts_centre", "casino", "cinema", "community_centre", "conference_centre", "fountain", "nightclub", "theatre", "place_of_worship",
        "courthouse", "fire_station", "police", "post_office", "townhall"
    ], 
    'leisure': [
        'dog_park', 'garden', 'park', 'playground'
    ],
    'shop': [
        "bakery", "butcher", "chocolate", "ice_cream", "pastry", "seafood", "tea", "supermarket", "kiosk", "convenience"
    ]
}

AMENITY_TYPES = {
    "Sustenance": [
        "bar", "biergarten", "cafe", "fast_food", "food_court", "ice_cream", "pub", "restaurant", 
        "bakery", "butcher", "chocolate", "ice_cream", "pastry", "seafood", "tea", "supermarket", "convenience"
    ],
    "Education": [
        "college", "driving_school", "kindergarten", "language_school", "library", "research_institute", "school", "music_school", "university"
    ], 
    "Transportation": [
        "fuel", "charging_station", "motorcycle_parking", "bicycle_parking", "parking", "taxi"
    ],
    "Financial": [
        "atm", "bank", "bureau_de_change", "money_transfer"
    ],
    "Healthcare": [
        "clinic", "dentist", "doctors", "hospital", "nursing_home", "pharmacy", "social_facility", "veterinary"
    ], 
    "Entertainment, Arts & Culture": [
        "arts_centre", "casino", "cinema", "community_centre", "conference_centre", "fountain", "nightclub", "theatre", "place_of_worship",
        "kiosk"
    ],
    "Public Service": [
        "courthouse", "fire_station", "police", "post_office", "townhall"
    ],
    "Green Spaces": [
        'dog_park', 'garden', 'park', 'playground'
    ]

}

def get_amenity_list():
    full_list = []
    for amenity_class in AMENITY_TYPES.keys():
        full_list += AMENITY_TYPES[amenity_class]
    return full_list


def get_amenity_class(amenity: str):
    for amenity_class in AMENITY_TYPES.keys():
        if amenity in AMENITY_TYPES[amenity_class]:
            return amenity_class
    return None

def download_pois(area_of_interest: shapely.Polygon):

    tags = POI_FILTER
    pois_df = ox.features_from_polygon(area_of_interest, tags)

    return pois_df

import numpy as np

def set_pois_to_edges(graph: nx.MultiDiGraph | nx.MultiGraph, pois_df):    
    nx.set_edge_attributes(graph, values=None, name="amenities")
    buffer_distance = 100 #meters
    buffer_meters = utils.m_to_deg(buffer_distance)

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges = edges.reset_index()

    for idx, poi in pois_df.iterrows():
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

            if graph[u][v][key]["amenities"] is None:
                graph[u][v][key]["amenities"] = set()

            if "amenity" in poi and poi["amenity"] is not np.nan:
                graph[u][v][key]["amenities"].add(poi["amenity"])
            if "leisure" in poi and poi["leisure"] is not np.nan:
                graph[u][v][key]["amenities"].add(poi["leisure"])
            if "shop" in poi and poi["shop"] is not np.nan:
                graph[u][v][key]["amenities"].add(poi["shop"])

    for u,v,key,data in graph.edges(data=True, keys=True):
        if graph[u][v][key]["amenities"] is None:
            graph[u][v][key]["amenities"] = []
        else:
            graph[u][v][key]["amenities"] = list(graph[u][v][key]["amenities"])

    return graph
