import json
import math
import numpy as np
import pandas as pd
import shapely
from geopy import distance as geo_distance
import networkx as nx

"""
General utility functions for the other modules
"""

DEG_CONVERT = 111595.75 

def to_wkt(geom: shapely.geometry.base.BaseGeometry) -> str:
    """
    Convert from Shapely geometry to string WKT (Well Known Text) representation.
    
    Parameters
    ----------
    point : shapely.geometry.base.BaseGeometry
            A Shapely BaseGeometry. Accepts any kind of geometry, such as Point, LineString, or Polygon.

    Returns 
    -------
    str
        The geometry in WKT representation (string).
    """
    if geom:
        return geom.wkt
    else: return None

def distance(p1:list[float,float], p2:list[float,float], is_geodetic=True) -> float:
    """
    Calculate distance between 2 points, in meters. Each point is represented as an array of 2 elements.
    
    Parameters
    ----------
    p1 : list[float,float]
        Second point in format [x,y].

    p2 : list[float,float]
        Second point in format [x,y].

    is_geodetic (optional) : boolean
        If this parameter is true, calculate the great arch distance (geodetic). If not, use euclidean distance.
        By default, the geodetic is calculated.

    Returns
    -------
    float
        The distance in decimal format. The distance represents meters, or the unit of measurement of the coordinate system, if calculated
        as euclidean distance.
    """
    dist = 0
    if is_geodetic:
        dist = geo_distance.distance(p1, p2).meters
    else:
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    return dist

def buffer(geom: shapely.LineString, buffer_size: float, erode_distance: float) -> shapely.Polygon:
    """
    Calculate an square buffer around a LineString geometry that cuts a distance equal to the erode_distance on each end of the
    line. A square buffer is a buffer that only expands the geometry over the line length, but not on the ends.
    
    This function is used to determine if a specific street is close to another one by a set buffer size. The erode distance allows to "cut"
    the ends of the line on the buffer to avoid intersection with adjacent streets.

    If the entire segment (geom) is shorter than twice the erosion distance, do not consider the erosion distance.

    Parameters
    ----------
    geom : shapely.LineString
        The geometry to which apply the buffer. The geometry must be a LineString.

    buffer_size : float
        The buffer size to apply.

    erode_distance : float
        The erosion distance to apply and cut on the ends of the line geometry.

    Returns
    -------
    shapely.Polygon
        A Polygon with the desired buffer and erosion distance.
    
    """
    # erode only if the geometries are big enough
    erode = False
    if geom.length > (erode_distance * 2):
        erode = True
        buffer_distance = buffer_size + erode_distance        
    else:
        erode = False
        buffer_distance = buffer_size
        # first do the buffer
    
    # first do the buffer
    buffered = geom.buffer(buffer_distance, cap_style="flat")

    # then erode to remove the edges that can interact with adjacent segments.
    # erosion is achieved by doing a negative buffer
    if erode:
        buffered = buffered.buffer(erode_distance * -1, cap_style="square")

    return buffered

def buffer_point(geom: shapely.Point, buffer_size: float) -> shapely.Polygon:
    """
    Calculate a buffer around a Point geometry.

    Parameters
    ----------
    geom : shapely.LineString
        The geometry to which apply the buffer. The geometry must be a Point.

    buffer_size : float
        The buffer size to apply in the measurement units of the CRS.

    Returns
    -------
    shapely.Polygon
        A Polygon with the desired buffer.
    
    """
    buffered = geom.buffer(buffer_size, quad_segs=16)
    return buffered

def intersects(buffer_geom: shapely.Polygon, line: shapely.LineString) -> bool:
    """
    Checks if the interesection between the polygon "buffer_geom" and the line "line" exists.

    Parameters
    ----------
    buffer_geom : shapely.Polygon
        A Polygon geometry

    line : shapely.LineString
        A LineString geometry

    Returns
    -------
    bool
        Whether the intersection between the line and polygon exists.
    """
    return line.intersects(buffer_geom)

def explore_graph(u:any, g:nx.Graph, depth:int) -> list[str]:
    """
    BFS-based (Breadth First Search) algorithm that returns a list of edges of a Graph with a maximum depth.
    It builds a subgraph of a set depth and returns its edges. The edges are bi-directional, meaning that for each 
    edge {u,v}, the correspondant edge {v,u} will also be added to the list.
    
    The edges are stored as strings with the following format 'u,v'. (Example: '7113200714,5592949065'). 
    This is due to the fact that sets cannot store arrays, as they are not hashable.

    The algorithm de-hashes the list of edges before returning, effectively yielding a list of arrays.

    Parameters
    ----------
    u : Initial node from which the BFS starts.
    g : An instance of a Networkx graph where the exploration will take place.
    depth : Maximum depth that the algorithm will explore. Higher depths takes more time and space.

    Returns
    -------
    list of tuples (u,v,key) of the edges of the subgraph. 

    """
    # A set avoids adding the same edge twice.
    edges_to_visit:set[str] = set()
    # u is the node id, d is the depth of the node.
    queue = [{"u":u, "d":0}]
    # Save the already visited nodes.
    visited = [u]
    while len(queue) > 0: 
        # extract and remove the first value of the list
        curr = queue.pop(0)
        curr_u = curr["u"]
        curr_d = curr["d"]
        v_list = g[curr_u]
        # iterate over the edges associated with node curr_u
        for curr_v in v_list:
            if curr_v != curr_u:
                # add u,v and also v,u.
                edges_to_visit.add(f"{curr_u},{curr_v}")
                edges_to_visit.add(f"{curr_v},{curr_u}")
            
            # avoid visiting the same node twice
            if curr_v not in visited:
                visited.append(curr_v)
                # control depth - do not add the node if the depth surpases the depth parameter
                new_d = curr_d + 1
                if new_d <= depth:
                    # Add the new node to explore it
                    queue.append({"u": curr_v, "d": new_d})

    edges_to_visit_list = list(map(lambda x: x.split(","), list(edges_to_visit)))
    return list(edges_to_visit_list)

