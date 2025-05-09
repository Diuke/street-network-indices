"""
NetworkExtractor is a suite of python functions for downloading realistic driving, pedestrian, cycling, and public transport street networks 
from OpenStreetMap for network analysis.

Networks are downloaded using OSMnx and postprocessed to reduce inconsistencies between the reality and OSM. 
"""
from __future__ import annotations

import json
import copy
import numpy as np
import networkx as nx
import osmnx as ox
from osmnx import io as ox_io
from osmnx import settings
import shapely
import ray
from modules.modes import pedestrian, cycling, driving, public_transport
from modules import assess

class NetworkExtractor():
    """
    NetworkExtractor is a suite of python functions for downloading realistic driving, pedestrian, cycling, and public transport street networks 
    from OpenStreetMap for network analysis.

    Networks are downloaded using OSMnx and postprocessed to reduce inconsistencies between the reality and OSM. 
    
    """

    DATA_BASE_PATH = "/home/user/Desktop/JP/street-network-indices/data"
    """
    Absolute base path where data is stored. All saving and loading operations are made relative to this path.
    """

    NODE_TAGS = ["ref", "highway", "crossing", "bicycle"]
    """
    Default custom node attributes to be downloaded from OSM. Modify to include more or less attributes.
    """

    WAY_TAGS = [
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
    """
    Default custom edge attributes to be downloaded from OSM. Modify to include more or less attributes.
    """

    def __init__(self):
        """
        Initialize the network extractor. 

        This network extractor overwrites some OSMnx functionalities. The initialization modifies the OSMNx settings.

        Loads the custom Node and Way tags on the OSMnx settings, 
        """
        settings.bidirectional_network_types = []
        self.DEFAULT_NODE_TAGS: list[str] = json.loads(json.dumps(settings.useful_tags_node))
        self.DEFAULT_WAY_TAGS: list[str] = json.loads(json.dumps(settings.useful_tags_way))
        for n_t in self.NODE_TAGS:
            if n_t not in settings.useful_tags_node:
                settings.useful_tags_node.append(n_t)
                    
        # for w_t in self.WAY_TAGS:
        #     if w_t not in settings.useful_tags_way:
        #         settings.useful_tags_way.append(w_t)
        settings.useful_tags_way = self.WAY_TAGS                       

    ####################
    # Public functions #
    ####################

    def extract_network(self, net_type: str, geometry: shapely.Polygon, options: dict = {}) -> nx.Graph:
        """
        Extract an OpenStreetMap street network within a specified geometry using OSMNx and post-processing.

        The networks that can be extracted are the driving ("drive"), pedestrian ("walk"), and cycling ("bike") networks, and are
        represented as a networkx Graph.

        The options parameter is used to send additional information to the specific 

        Parameters
        ----------
        net_type (str) : 
            Type of network to extract. It can be one of the following:

            "walk" | "bike" | "drive" | "public_transport"

            Each network provides different a different processing to build realistic street networks.

        geometry (shapely.Poligon) : 
            Geometry from which the network will be extracted. The format is a shapely Polygon.

        options (dict) : 
            Optional parameter that allows to send data to the preprocessing methods of transportation mode. The options are the following:

            - assessment: Parameter for "walk" networks. Avoid the elimination of duplicated paths and add a flag for further assessment.
            - dist_threshold: Custom distance threshold for duplicate pedestrian path elimination. If not specified, 20 meters threshold is used.
            - slope_threshold: Custom bearing angle threshold for duplicate pedestrian path elimination. If not specified, 15 degrees of threshold is used.

        Returns
        -------
        A networkx.MultiGraph or networkx.MultiDiGraph, depending on the type of network.

        A non directed Graph (MultiGraph) representation is used for pedestrian networks, as direction is not important.

        A directed Graph (MultiDiGraph) representation is used for driving, cycling, and public transport networks, as direction is important.
        """ 
        network_type = net_type

        # Select the custom filter, if available, for each type of transportation mode
        graph_edge_tags = self.WAY_TAGS
        if net_type == "walk":
            custom_filter = pedestrian.CUSTOM_WALK_FILTER
            graph_node_tags = pedestrian.PEDESTRIAN_GRAPH_NODE_TAGS
            graph_edge_tags = pedestrian.PEDESTRIAN_GRAPH_EDGE_TAGS
        elif net_type == "bike":
            custom_filter = cycling.CUSTOM_CYCLING_FILTER
            graph_node_tags = cycling.CYCLING_GRAPH_NODE_TAGS
            graph_edge_tags = cycling.CYCLING_GRAPH_EDGE_TAGS
        elif net_type == "drive": 
            custom_filter = None
            graph_node_tags = pedestrian.PEDESTRIAN_GRAPH_NODE_TAGS
            graph_edge_tags = pedestrian.PEDESTRIAN_GRAPH_EDGE_TAGS
        elif net_type == "public_transport": 
            custom_filter = None
        else:
            # The network type is not supported.
            raise Exception("Network type not supported")
        
        # avoid cache problems by disabling it
        settings.use_cache = False

        # Public transport network download is "different".
        if net_type == "public_transport":
            clean_g = public_transport.process_public_transport_graph(geometry)
            graph_metadata = {
                "created_date": ox.utils.ts(),
                "created_with": f"OSMnx {ox.__version__}",
                "crs": settings.default_crs,
            }

        # For driving, cycling, and pedestrian networks.
        else:
            # set the tags for download
            for n_t in self.NODE_TAGS:
                if n_t not in settings.useful_tags_node:
                    settings.useful_tags_node.append(n_t)
            
            # Set the custom tags for OSMnx.
            settings.useful_tags_way = graph_edge_tags
            
            # Use the OSMnx graph_from_polygon functionality to download a preliminary network with the custom filter
            g = ox.graph_from_polygon(
                geometry,
                network_type=network_type, 
                custom_filter=custom_filter,
                simplify=False,
                retain_all=True,
                truncate_by_edge=True
            )
            graph_metadata = copy.deepcopy(g.graph)

            # Report some stats
            print("finish download")
            print(f"Raw number of edges: {g.number_of_edges()}")
            print(f"Raw number of nodes: {g.number_of_nodes()}")

            # The graph is directed by default (MultiDiGraph)
            # Convert to GeoDataFrame for filtering
            gdf = ox.graph_to_gdfs(g)

            if net_type == "walk":
                # Load options for walk
                assessment = options["assessment"] if "assessment" in options else False
                dist_threshold = options["dist_threshold"] if "dist_threshold" in options else None
                slope_threshold = options["slope_threshold"] if "slope_threshold" in options else None

                # Extract the pedestrian network graph
                clean_g = pedestrian.process_pedestrian_graph(gdf, assessment=assessment, dist_threshold=dist_threshold, slope_threshold=slope_threshold)

            elif net_type == "drive":
                # Extract the driving network graph
                clean_g = driving.process_driving_graph(gdf)

            elif net_type == "bike":
                # Extract the cycling network graph
                clean_g = cycling.process_cycling_graph(gdf)

            else:
                raise Exception("Network Type Not Supported")
                    
        # restore original graph metadata
        clean_g.graph = graph_metadata
        print("finish processing")
        print(f"Final number of edges: {clean_g.number_of_edges()}")
        print(f"Final number of nodes: {clean_g.number_of_nodes()}")
        return clean_g
        
    def save_as_shp(self, g:nx.Graph, path:str, save_nodes=True, save_edges=True, line_graph=False, node_attributes=None, edge_attributes=None):
        """
        Save a osmnx as a Shapefile for visualization in GIS software.

        Parameters
        ----------
        g (networkx.Graph) : 
            The graph to save as shapefile

        path (str) :
            Relative path with respect to the DATA_BASE_PATH in which the shapefile will be saved.

        save_nodes (boolean) :
            Optional boolean parameter to save a shapefile containing the nodes of the graph. It will save a Point geometry
            if the graph is primal, and LineString geometry in case it is a line graph. It is True by default.

        save_edges (boolean) : 
            Optional boolean parameter to save a shapefile containing the edges of the graph. It will save a LineString geometry
            if the graph is primal, and Point geometry in case it is a line graph. It is True by default.

        line_graph (boolean) :
            Optional parameter to specify if the graph is a Line Graph. Line graphs are a network representation where intersections 
            are represented as edges and street network segments are represented as nodes. It is False by default, but if sent as True,
            it will save nodes as LineStrings and edges as Points. 
        """
        if line_graph:
            nodes, edges = ox.graph_to_gdfs(g, node_geometry=False, fill_edge_geometry=False)
            nodes.crs = "epsg:4326"
            edges.crs = "epsg:4326"
        else:
            # saving the graph nodes and edges 
            nodes, edges = ox.graph_to_gdfs(g)

            if "osmid" in list(nodes.columns):
                nodes = nodes.reset_index(drop=True)
            else:
                nodes = nodes.reset_index(drop=False)

            nodes = nodes.replace(np.nan, None)

            if "osmid" in list(edges.columns):
                edges = edges.reset_index(drop=True)
            else:
                edges = edges.reset_index(drop=False)

            edges = edges.replace(np.nan, None)

        if node_attributes is not None:
            node_columns = node_attributes
        else: 
            node_columns = nodes.columns

        if edge_attributes is not None:
            edge_columns = edge_attributes
        else: 
            edge_columns = edges.columns

        # Encoding the nodes of any object-based column that is not a primitive.
        for col in node_columns:
            if nodes[col].dtype == object:
                nodes[col] = nodes[col].astype(str)

        # Encoding the edges of custom properties.
        for col in edge_columns:
            print(col)
            if edges[col].dtype == object:
                edges[col] = edges[col].astype(str)
            if col == "key":
                edges[col] = edges[col].astype(np.int16)
            if col == "speed_kph":
                edges[col] = edges[col].astype(np.int16)
            if col == "travel_time":
                edges[col] = edges[col].astype(np.float32)
            if col == "grade":
                edges[col] = edges[col].astype(np.float32)
            if col == "grade_abs":
                edges[col] = edges[col].astype(np.float32)
            if col == "length":
                edges[col] = edges[col].astype(np.float32)           
                
        if save_nodes:
            nodes.to_file(f"{self.DATA_BASE_PATH}/{path}_nodes.shp", encoding='utf-8')  
        if save_edges:
            edges.to_file(f"{self.DATA_BASE_PATH}/{path}_edges.shp", encoding='utf-8')  

    def save_as_graph(self, g: nx.Graph, path: str):
        """
        Save the received graph using the GraphML format. 

        Parameters
        ----------
        g (networkx.Graph) :
            The graph to save as GraphML

        path (str) :
            Relative path with respect to the DATA_BASE_PATH in which the shapefile will be saved.
        """
        ox_io.save_graphml(g, f'{self.DATA_BASE_PATH}/{path}.graphml')
        return

    def plot_graph(self, g: nx.Graph):
        """
        Plot a graph using osmnx and the function plot_graph().
        """
        ox.plot_graph(g)

    def __deserialize_point_string(self, point_string: str) -> shapely.Point :
        """ 
        Private method used for deserializing a Point string, which is stored as string when serialized.
        """
        return shapely.from_wkt(point_string)
    
    def __deserialize_bearing(self, bearing: None) -> shapely.Point :
        """ 
        Private method used for deserializing the bearing angle of streets.
        It is necessary to create this function, as a None bearing is saved to file as a string.
        """
        if bearing == "None": return None
        else: return float(bearing)

    def load_graph(self, path: str) -> nx.MultiGraph | nx.MultiDiGraph:
        """
        Load a graph in GraphML format using custom deserializers.

        This function allows to retrieve previously saved graphs that were saved using the save_as_graph 
        """
        node_types = {
            "geometry": self.__deserialize_point_string
        }
        edge_types = {
            "bearing": self.__deserialize_bearing
        }
        return ox_io.load_graphml(
            f'{self.DATA_BASE_PATH}/{path}.graphml', 
            edge_dtypes=edge_types,
            node_dtypes=node_types
        )          
    

    @ray.remote
    def download_network(self, 
                         network_type:str, 
                         geometry, 
                         city_name, 
                         assessment=False, 
                         dist_threshold=None, 
                         slope_threshold=None,
                         add_elevation=False,
        ) -> nx.MultiDiGraph | nx.MultiGraph:
        """
        Ray-powered paralellized function for downloading a street network from a geometry.

        It can be called using:

        .. code-block:: text
            ray.get()

        Parameters
        ----------
        network_type (str) :
            The network type to dowload. It can be "drive", "walk", "bike", or "public_transport".

        geometry (shapely.Polygon) :
            A shapely polygon from which the OSM network will be downloaded.

        city_name (str) :
            The name of the city to download. It will be used for extracting the DEM information if add_elevation is True.
        
        assessment (bool):
            Optional. Enable assessment of duplicated pedestrian streets. When True, it does not remove any street, 
            but adds the property "assess" based on possible duplicated pedestrian streets.

        dist_threshold (int | None): 
            Optional. The buffer distance to apply to driving streets for checking duplicated pedestrian sidewalks.
            By default is 20 meters.

        slope_threshold (int | None): 
            Optional. The maximum angle of difference between driving and pedestrian streets for checking duplicated pedestrian sidewalks.
            By default is 15 degrees.

        add_elevation (bool):
            Optional. When True adds the elevation property to each node. The elevation is calculated using a DEM which must be 
            located inside the base data path in a folder called DEM. The DEM must have the name {city_name}_DEM.tif.
            This property is used for calculating the slope of each street segment.


        
        """
        print(f"extracting {network_type} for {city_name}")

        options = {"assessment": assessment, "dist_threshold": dist_threshold, "slope_threshold": slope_threshold}
        g = self.extract_network(network_type, geometry=geometry, options=options)
        
        if add_elevation:
            DEM_location = f'{self.DATA_BASE_PATH}/DEM/{city_name}_DEM.tif'
            ox.add_node_elevations_raster(g, DEM_location, cpus=4)
            ox.add_edge_grades(g)

        return g
    
    def download_network_sync(self, 
                        network_type:str, 
                        geometry, 
                        city_name, 
                        assessment=False, 
                        dist_threshold=None, 
                        slope_threshold=None,
                        add_elevation=False,
        ) -> nx.MultiDiGraph | nx.MultiGraph:
        """
        Download a street network from a Geometry in a synchronous way. Ideal for downloading only one network at the time.

        For a faster download of multiple networks at the same time, it is preferred to use the paralellized function download_network.

        Parameters
        ----------
        network_type  (str) :
            The network type to dowload. It can be "drive", "walk", "bike", or "public_transport".

        geometry (shapely.Polygon): 
            A shapely polygon from which the OSM network will be downloaded.

        city_name : The name of the city or settlement to download. Used for extracting the DEM if the parameter add_elevation is True.
        assessment : Optional. Enable assessment of duplicated pedestrian streets. When True, it does not remove any street, but adds the property "assess" based on possible duplicated pedestrian streets.
        dist_threshold : Optional. The buffer distance to apply to driving streets for checking duplicated pedestrian sidewalks.
        slope_threshold : Optional. The maximum angle of difference between driving and pedestrian streets to be considered 

        
        """
        options = {"assessment": assessment, "dist_threshold": dist_threshold, "slope_threshold": slope_threshold}
        g = self.extract_network(network_type, geometry=geometry, options=options)
        
        if add_elevation:
            DEM_location = f'{self.DATA_BASE_PATH}/DEM/{city_name}_DEM.tif'
            ox.add_node_elevations_raster(g, DEM_location, cpus=4)
            ox.add_edge_grades(g)

        return g
    
    def assess_network(self, net_type: str, geometry: shapely.Polygon, options: dict = {}) -> nx.MultiDiGraph | nx.MultiGraph:
        """
        Download and assess a pedestrian or cycling network.
        """
        
        # Pedestrian or Cycling
        network_type = net_type

        # Select the custom filter, if available, for each type of transportation mode
        graph_edge_tags = self.WAY_TAGS
        if net_type == "walk":
            custom_filter = pedestrian.CUSTOM_WALK_FILTER
            graph_node_tags = pedestrian.PEDESTRIAN_GRAPH_NODE_TAGS
            graph_edge_tags = pedestrian.PEDESTRIAN_GRAPH_EDGE_TAGS
        elif net_type == "bike":
            custom_filter = cycling.CUSTOM_CYCLING_FILTER
            graph_node_tags = cycling.CYCLING_GRAPH_NODE_TAGS
            graph_edge_tags = cycling.CYCLING_GRAPH_EDGE_TAGS
        else:
            # The network type is not supported.
            raise Exception("Network type not supported")
        
        # avoid cache problems by disabling it
        settings.use_cache = False

        # set the tags for download
        for n_t in self.NODE_TAGS:
            if n_t not in settings.useful_tags_node:
                settings.useful_tags_node.append(n_t)
        
        # Set the custom tags for OSMnx.
        settings.useful_tags_way = graph_edge_tags
        
        # Use the OSMnx graph_from_polygon functionality to download a preliminary network with the custom filter
        g = ox.graph_from_polygon(
            geometry,
            network_type=network_type, 
            custom_filter=custom_filter,
            simplify=False,
            retain_all=True,
            truncate_by_edge=True
        )
        graph_metadata = copy.deepcopy(g.graph)

        # Report some stats
        print("finish download")
        print(f"Raw number of edges: {g.number_of_edges()}")
        print(f"Raw number of nodes: {g.number_of_nodes()}")

        # The graph is directed by default (MultiDiGraph)
        # Convert to GeoDataFrame for filtering
        gdf = ox.graph_to_gdfs(g)

        if net_type == "walk":
            # Extract the pedestrian network graph
            clean_g = assess.assess_pedestrian_graph(gdf, simplify=True)
            
            #clean_g = pedestrian.process_pedestrian_graph(gdf, assessment=assessment, dist_threshold=dist_threshold, slope_threshold=slope_threshold)

        elif net_type == "bike":
            # Extract the cycling network graph
            clean_g = cycling.process_cycling_graph(gdf, assessment=False)

        else:
            raise Exception("Network Type Not Supported")
                    
        # restore original graph metadata
        clean_g.graph = graph_metadata
        print("finish processing")
        print(f"Final number of edges: {clean_g.number_of_edges()}")
        print(f"Final number of nodes: {clean_g.number_of_nodes()}")
        return clean_g