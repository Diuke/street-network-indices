import copy
import networkx as nx
import osmnx as ox
from modules.generalize import Generalize
from modules.network_extractor import NetworkExtractor

class Networker():
    layers: list[str]
    graphs: dict[str, nx.Graph]

    REPRESENTATION_TYPES = ["complete", "topology", "raster"]
    # Representation types:
    #   complete    -> MultiDiGraph, direction is important. Default
    #   topology    -> Undirected Graph. Only shape is important.
    #   raster      -> Rasterize the graph geometry.

    def __init__(self):
        # self.graph = nx.MultiDiGraph()
        self.extractor = NetworkExtractor()
        self.layers = []
        self.graphs = {}
        self.generalizer = Generalize()

    def check_edge_exists(self, g:nx.Graph, u, v):
        if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
            return g.has_edge(u, v, self.key)
        else:
            return g.has_edge(u, v)
        
    def add_edge(self, g:nx.Graph, u, v, data={}):
        if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
            g.add_edge(u, v, self.key, **data)
        else:
            g.add_edge(u, v, **data)

    def add_node(self, g:nx.Graph, node_id, data={}):
        g.add_node(node_id, **data)
        
    def get_edge_property(self, g:nx.Graph, u, v, property):
        if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
            return g[u][v][self.KEY][property]
        else:
            return g[u][v][property]

    def update_edge_property(self, g:nx.Graph, u, v, property, value):
        if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
            g[u][v][self.KEY][property] = value
        else:
            g[u][v][property] = value

    def get_node_property(self, g:nx.Graph, node_id, property):
        return g._node[node_id][property]

    def update_node_property(self, g:nx.Graph, node_id, property, value):
        g._node[node_id][property] = value

    def __merge_layers(self, layers: list[str], graphs: dict[str, nx.Graph]):
        graph = nx.Graph()

        for name in layers:
            g = copy.deepcopy(graphs[name])
            # Set the CRS of the graph using the first added graph
            if "crs" not in graph.graph:
                if "crs" in g.graph:
                    graph.graph["crs"] = g.graph["crs"]
                else: 
                    graph.graph["crs"] = "epsg:4326"

            try:
                #TODO add the layer to the graph
                # save the original graph
                add_nodes = g.nodes(data=True)
                for node in add_nodes:
                    add_node_id = node[0]
                    if not graph.has_node(add_node_id):

                        node_data = copy.deepcopy(node[1])
                        # add the current transport mode to the node
                        node_data["modes"] = [name]
                        self.add_node(graph, add_node_id, data=node_data)
                    else: 
                        # add the transport mode to the existing node
                        node_modes_list = self.get_node_property(graph, add_node_id, "modes")
                        if name not in node_modes_list:
                            node_modes_list.append(name)
                        self.update_node_property(graph, add_node_id, "modes", node_modes_list)

                add_edges = g.edges(data=True)
                for edge in add_edges:
                    add_edge_u = edge[0]
                    add_edge_v = edge[1]

                    if not self.check_edge_exists(graph, add_edge_u, add_edge_v):
                        edge_data = copy.deepcopy(edge[2])
                        # add the current transport mode to the node
                        edge_data["modes"] = [name]
                        self.add_edge(graph, add_edge_u, add_edge_v, data=edge_data)
                    else: 
                        # add the transport mode to the existing node
                        modes_list = self.get_edge_property(graph, add_edge_u, add_edge_v, "modes")
                        if name not in modes_list:
                            modes_list.append(name)
                        self.update_edge_property(graph, add_edge_u, add_edge_v, "modes", modes_list)
            
            except Exception as ex:
                raise ex
        
        return graph
        
    def add_layer(self, g: nx.Graph, name: str):
        if name in self.layers:
            print(f"Layer {name} already exists")
            raise Exception(f"Layer {name} already exists")

        else:
            self.layers.append(name)
            self.graphs[name] = g
        

    def transform(self, to: str):
        # Transformations are possible from:
        #   complete -> topology
        #   topology -> complete
        #   complete -> raster
        copy_g = copy.deepcopy(self.graph)

        if to in self.REPRESENTATION_TYPES:
            # transform to specific format
            if to == "complete":
                converted = copy_g
            elif to == "topology":
                # remove duplicated edges
                converted = nx.MultiDiGraph()
                converted.graph["crs"] = self.graph.graph["crs"]
                for node, data in copy_g.nodes(data=True):
                    if not converted.has_node(node):
                        self.add_node(converted, node, data=data)

                for u,v,data in copy_g.edges(data=True):
                    if not self.check_edge_exists(converted, u, v) and not self.check_edge_exists(converted, v, u):
                        self.add_edge(converted, u, v, data=data)
                return converted                   

            else:  #raster
                # ???????
                pass

        return converted
    
    def get_layer_names(self) -> list[str]:
        """
        Get a list of the layer names
        """
        return self.layers
    
    def get_graph(self, type:str="complete", dual:bool=False, rasterize:bool=False, simplification:str|None=None, layers:list[str]|None=None, lod:int=10):
        # type           : The type of graph to get -> complete (directed) or topological (undirected)
        # dual           : false for primal graph, true for dual graph
        # rasterize      : true for raster, false for normal graph
        # simplification : simplification modes... (preserve_topology | natural_streets)
        
        final_graphs = {}        
        if layers is None:
            layers = self.layers
        
        for layer in layers:
            if layer not in self.layers:
                raise Exception("Layer not found in graphs.")

            g = copy.deepcopy(self.graphs[layer])
            # First do conversion to the specific type
            if type == "complete":
                # do nothing, as it is already the complete graph
                return_graph = copy.deepcopy(g)
            else:
                # transform
                return_graph = self.transform(g, "topology")

            # Third apply simplification
            if simplification is not None:
                if simplification == "preserve_topology":
                    return_graph = self.generalizer.topology_preservation_generalization(
                        input_graph=return_graph,
                        max_iterations=40
                    )
                elif simplification == "natural_streets":
                    return_graph = self.generalizer.named_streets_generalization(
                        input_graph=return_graph
                    )

                else:
                    raise Exception("Invalid simplification algorithm")
                
            # Second calculate dual, if requested
            if dual:
                return_graph = nx.line_graph(return_graph)
            
            final_graphs[layer] = copy.deepcopy(return_graph)
        
            #TODO: Add elevations to nodes!
            #ox.add_node_elevations_raster
        
        if len(layers) > 1:
            # Merge layers
            return_graph = self.__merge_layers(layers, final_graphs)
        else:
            # There is only one layer, return it
            return_graph = final_graphs[layers[0]]

        if not return_graph.is_multigraph():
            if return_graph.is_directed():
                return_graph = nx.MultiDiGraph(return_graph)
            else:
                return_graph = nx.MultiGraph(return_graph)

        return return_graph

    def remove_layer(self, name: str):
        # Check first if layer exist
        if name not in self.layers:
            print(f"Layer {name} does not exist")
            raise Exception(f"Layer {name} does not exist")

        else:
            # Remove the name of the layer from the list
            self.layers.remove(name)
            del self.graphs[name]

    def get_layer(self, name: str):
        if name in self.graphs.keys():
            return self.graphs[name]
        else:
            raise Exception(f"Layer {name} does not exist in graph")
        
    def add_elevation(self, graph: nx.Graph, city_name: str, cpus=1):
        DEM_location = f'{self.extractor.DATA_BASE_PATH}/DEM/{city_name}_DEM.tif'
        ox.add_node_elevations_raster(graph, DEM_location, cpus=cpus)
