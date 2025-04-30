import copy
import networkx as nx
import osmnx as ox
from shapely import ops, LineString
from itertools import chain
from modules import utils
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

ANGLE_THRESHOLD = 10

def is_insterstitial(node_id, g: nx.MultiDiGraph | nx.MultiGraph):
    # for directed, an interstitial node has 1 incomming and 1 outgoing edges
    if g.is_directed():
        if g.out_degree(node_id) == 1 and g.in_degree(node_id) == 1:
            return True
        
    # for undirected, an interstitial node exactly 2 edges
    else:
        if g.degree(node_id) == 2:
            return True
    
    # in any othe cases, return false
    return False

def merge_intersections(input_g: nx.MultiDiGraph | nx.MultiGraph, tolerance=10):
    """
    Use the OSMNX consolidate intersections algorithm for simplifying the intersections of the graph.
    """
    tolerance = tolerance / utils.DEG_CONVERT
    simplified = ox.simplification.consolidate_intersections(input_g, tolerance=tolerance, rebuild_graph=True)
    return simplified

def recursive_find_natural_street(input_g:nx.MultiGraph, u, v, key, prev=None, edges=[], visited:set=set()):
    """
    Simplification of natural street calculation.

    input_g is the entire graph to be analyzed
    prev is the node from which the recursion is comming. None when is the first edge.
    u is the node u of the edge to analyse
    v is the node v of the edge to analyse
    edges is the list of the edges of the natural street so far codified as strings 'u,v'
    visited is the set of visited edges codified as string 'u,v' so it does not visit again certain edges.
    """

    # It is undirected in the end...
    if f"{u},{v},{key}" in visited or f"{v},{u},{key}" in visited:
        return edges, visited 
    else:
        visited.add(f'{u},{v},{key}')
    
    if len(edges) == 0:
        edges = set()
        edges.add(f'{u},{v},{key}')

    edge_data = input_g[u][v][key]

    adjacent_u = []
    if prev != u:
        adjacent_u = chain(nx.bfs_predecessors(input_g, u, depth_limit=1), nx.bfs_successors(input_g, u, depth_limit=1))

    adjacent_v = []
    if prev != v:
        adjacent_v = chain(nx.bfs_predecessors(input_g, v, depth_limit=1), nx.bfs_successors(input_g, v, depth_limit=1))

    if "name" in edge_data:
        street_name = edge_data["name"]
    else:
        street_name = None
    
    for node_id, adjacent_nodes in adjacent_u:
        
        if type(adjacent_nodes) is not list:
            adjacent_nodes = [adjacent_nodes]
        for adj_node in adjacent_nodes:
            if input_g.has_edge(node_id, adj_node):
                sub_u = node_id
                sub_v = adj_node
            else:
                sub_u = adj_node
                sub_v = node_id
            try:
                adj_edge_dict = input_g.get_edge_data(sub_u, sub_v)
            except:
                return edges, visited
            
            for adj_key, adj_edge_data in adj_edge_dict.items():
                check_angles = False
                if street_name is not None and street_name != "None":
                    if "name" in adj_edge_data and street_name == adj_edge_data["name"]:
                        # check for u,v and v,u in edges
                        if f'{sub_u},{sub_v},{adj_key}' not in edges and f'{sub_v},{sub_u},{adj_key}' not in edges:
                            edges.add(f'{sub_u},{sub_v},{adj_key}')
                            edges, visited = recursive_find_natural_street(input_g, sub_u, sub_v, adj_key, prev=u, edges=edges, visited=visited)
                    else: check_angles = True
                else: check_angles = True
                        
                if check_angles:
                    if "bearing" in adj_edge_data and "bearing" in edge_data and adj_edge_data["bearing"] is not None and edge_data["bearing"] is not None:
                        adj_angle = adj_edge_data["bearing"]
                        current_angle = edge_data["bearing"]
                        if (
                            (abs(adj_angle - current_angle) < ANGLE_THRESHOLD) or
                            (abs(((adj_angle + 180) % 360) - current_angle) < ANGLE_THRESHOLD)
                        ):
                            if f'{sub_u},{sub_v},{adj_key}' not in edges and f'{sub_v},{sub_u},{adj_key}' not in edges:
                                edges.add(f'{sub_u},{sub_v},{adj_key}')
                                edges, visited = recursive_find_natural_street(input_g, sub_u, sub_v, adj_key, prev=u, edges=edges, visited=visited)
    
    for node_id, adjacent_nodes in adjacent_v:
        if type(adjacent_nodes) is not list:
            adjacent_nodes = [adjacent_nodes]

        for adj_node in adjacent_nodes:
            if input_g.has_edge(node_id, adj_node):
                sub_u = node_id
                sub_v = adj_node
            else:
                sub_u = adj_node
                sub_v = node_id

            try:
                adj_edge_dict = input_g.get_edge_data(sub_u, sub_v)
            except:
                return edges, visited

            for adj_key, adj_edge_data in adj_edge_dict.items():
                check_angles = False
                if street_name is not None:
                    if "name" in adj_edge_data and street_name == adj_edge_data["name"]:
                        if f'{sub_u},{sub_v},{adj_key}' not in edges and f'{sub_v},{sub_u},{adj_key}' not in edges:
                            edges.add(f'{sub_u},{sub_v},{adj_key}')
                            edges, visited = recursive_find_natural_street(input_g, sub_u, sub_v, adj_key, prev=v, edges=edges, visited=visited)
                    else: check_angles = True
                else: check_angles = True
                        
                if check_angles:
                    if "bearing" in adj_edge_data and "bearing" in edge_data and adj_edge_data["bearing"] is not None and edge_data["bearing"] is not None:
                        adj_angle = adj_edge_data["bearing"]
                        current_angle = edge_data["bearing"]
                        if (
                            (abs(adj_angle - current_angle) < ANGLE_THRESHOLD) or
                            (abs(((adj_angle + 180) % 360) - current_angle) < ANGLE_THRESHOLD)
                        ):
                            if f'{sub_u},{sub_v},{adj_key}' not in edges and f'{sub_v},{sub_u},{adj_key}' not in edges:
                                edges.add(f'{sub_u},{sub_v},{adj_key}')
                                edges, visited = recursive_find_natural_street(input_g, sub_u, sub_v, adj_key, prev=v, edges=edges, visited=visited)

    return edges, visited

def named_streets_generalization(input_graph: nx.MultiDiGraph | nx.MultiGraph, cut_level:int=10, is_directed=False):
    # named STREETS SIMPLIFICATION
    input_g = copy.deepcopy(input_graph)

    # if the graph is directed, make it undirected
    if is_directed:
        input_g = input_g.to_undirected()

    # remove single disconnected edges
    remove_singles = []
    for u,v,key in input_g.edges(keys=True):
        if len(input_g[u]) <= 1 and len(input_g[v]) <= 1:
            remove_singles.append([u,v,key])
    input_g.remove_edges_from(remove_singles)

    input_g_copy = copy.deepcopy(input_g)
    edges_list = copy.deepcopy(input_g.edges(keys=True))

    print(f"initial edges: {len(edges_list)}")

    print("Starting recursive algorithm to find natural streets")
    named_streets = {}
    named_st_index = 0
    i = 0
    for u,v,key in edges_list:
        visited = set()
        if input_g_copy.has_edge(u,v,key):
            edges, _ = recursive_find_natural_street(input_g_copy, u, v, key, prev=None, visited=visited)
            if len(edges) > 0:
                named_streets[named_st_index] = []
                for temp_edge in list(edges):
                    uvkey = temp_edge.split(",")
                    temp_u = int(uvkey[0])
                    temp_v = int(uvkey[1])
                    temp_key = int(uvkey[2])
                    named_streets[named_st_index].append([temp_u,temp_v,temp_key])

                    if input_g_copy.has_edge(temp_u,temp_v,temp_key):
                        input_g_copy.remove_edge(temp_u,temp_v,temp_key)
                    elif input_g_copy.has_edge(temp_v,temp_u,temp_key):
                        input_g_copy.remove_edge(temp_v,temp_u,temp_key)

                named_st_index += 1

        i += 1
        if i % 10000 == 0: print(i)

    print("Finished recursive algorithm to find natural streets")

    # named_streets variable contains a dictionary with ID and the list of (u,v,key) edges that belong to the named street.
    
    # iterate the named streets and extract geometries, length, and nodes from the input graph.
    named_streets_info: dict[str,dict[str,list]] = {}
    print(f"Total streets {len(named_streets.items())}")

    # specify to which named streets each node belongs to
    node_to_named_street: dict[any, list] = {}

    # first iteration to assign named streets to nodes and to build geometries and lengths
    for named_street_id, named_street_edge_list in named_streets.items():
        named_streets_info[named_street_id] = {
            "geometries": [],
            "length": 0,
            "edges": [],
            "name": []
        }

        for u,v,key in named_street_edge_list:
            edge_data = input_g.get_edge_data(u,v,key)
            # street geometries
            named_streets_info[named_street_id]["geometries"].append(
                edge_data["geometry"]
            )
            # street length
            named_streets_info[named_street_id]["length"] += edge_data["length"]
            # street names
            edge_name = edge_data["name"] if "name" in edge_data else None
            if edge_name not in named_streets_info[named_street_id]["name"]:
                named_streets_info[named_street_id]["name"].append(edge_name)
            

            # add the named street to both u and v nodea if it does not exist
            if u not in node_to_named_street:
                node_to_named_street[u] = []
            node_to_named_street[u].append(named_street_id)
            
            if v not in node_to_named_street:
                node_to_named_street[v] = []
            node_to_named_street[v].append(named_street_id)

    # build the graph with the geometries
    named_streets_nodes = []
    named_streets_edges = {}
    i = 0
    for named_street_id, named_street_edge_list in named_streets.items():
        i += 1
        if i % 1000 == 0: print(i)
        # build the geometry and the edges from the named street
        # merge the line strings to a multiline geometry
        named_street_geometry = ops.linemerge(named_streets_info[named_street_id]["geometries"])
        named_street_centroid = named_street_geometry.centroid
        # assign the data to the new node (the named street)
        data = {
            "geometry": named_street_geometry, # LineString geometry
            "street_name": named_streets_info[named_street_id]["name"], # name(s) of the street
            "lenght": named_streets_info[named_street_id]["length"], # aggregated length (in meters)
            "segment_count": len(named_street_edge_list),
            "x": named_street_centroid.coords.xy[0][0],
            "y": named_street_centroid.coords.xy[1][0],
        }
        
        named_streets_edges[named_street_id] = set()
        for u,v,key in named_street_edge_list:
            u_named_street = node_to_named_street[u]
            v_named_street = node_to_named_street[v]

            for connected in u_named_street:
                if connected != named_street_id:
                    named_streets_edges[named_street_id].add(connected)

            for connected in v_named_street:
                if connected != named_street_id:
                    named_streets_edges[named_street_id].add(connected)         

        # Create a dict with the named street name to the node data
        if named_street_id not in named_streets_nodes:
            named_streets_nodes.append([named_street_id, copy.deepcopy(data)])

    # Build the named streets graph
    graph_metadata = copy.deepcopy(input_g.graph)
    named_streets_graph = nx.MultiGraph()
    # add nodes
    for node_id, data in named_streets_nodes:
        if not named_streets_graph.has_node(node_id):
            named_streets_graph.add_node(node_id, **data)

    # add edges
    for u, v_list in named_streets_edges.items():
        for v in v_list:
            if v != u:
                if not named_streets_graph.has_edge(u,v):
                    u_data = named_streets_graph._node[u]
                    v_data = named_streets_graph._node[v]
                    edge_data = {
                        "geometry": LineString([
                            [u_data["x"], u_data["y"]],
                            [v_data["x"], v_data["y"]],
                        ])
                    }
                    named_streets_graph.add_edge(u,v, **edge_data)
    # set the CRS to save as OSMnx-enabled graph 
    named_streets_graph.graph["crs"] = "epsg:4326"

    # remove unimportant nodes based on cut_level
    print("finished natural streets processing")
    named_streets_graph.graph = graph_metadata
    return named_streets_graph

def topology_preservation_generalization(input_graph: nx.MultiGraph | nx.MultiDiGraph, max_iterations=10):
    """
    Based on the generalization methodology propsed in DOI: 10.1007/s41109-022-00521-8 .

    It is an iterative algorithm that generalizes a traffic network by eliminating parallel edges,
    self-loops, dead-ends, low-level gridirons, and interstitial nodes (to be implemented).

    Works both for directed and undirected graphs.
    """

    graph_metadata = copy.deepcopy(input_g.graph)
    input_g = copy.deepcopy(input_graph)

    is_modified = True
    iterations = 0

    # add aggr_node_number property to all nodes
    for node_id in input_g.nodes():
        input_g._node[node_id]["aggr_node_number"] = 1

    print(f"Initial: {input_g}")
    # Iterate when the graph has been modified (it has not yet converged) or the max iterations has been reached.
    while is_modified and iterations <= max_iterations:
        is_modified = False
        iterations += 1
        print(f"iteration {iterations}")
        print(f"Starting Nodes: {input_g.number_of_nodes()}")
        print(f"Starting Edges: {input_g.number_of_edges()}")
        # 1) Parallel edges
        delete_edges = []
        for node in input_g.nodes():
            for node_to in input_g[node]:

                # select the shortest parallel edge
                if len(list(input_g[node][node_to].keys())) > 1:
                    min_parallel = 99999999
                    min_parallel_key = -1
                    for parallel_edge_key in input_g[node][node_to].keys():
                        data = input_g.get_edge_data(node, node_to, parallel_edge_key, default=None)
                        if data["length"] > min_parallel:
                            min_parallel = data["length"]
                            min_parallel_key = parallel_edge_key
                        
                    for parallel_edge_key in input_g[node][node_to].keys():
                        if parallel_edge_key != min_parallel_key:
                            delete_edges.append([node, node_to, parallel_edge_key])

        if len(delete_edges) > 0:
            is_modified = True
            print("removing parallel edges")
            input_g.remove_edges_from(delete_edges)


        # 2) Self loops
        delete_edges = []
        for u,v,key in input_g.edges(keys=True):
            if u == v:
                # self loop identified
                delete_edges.append([u,v,key])
                
        if len(delete_edges) > 0:
            is_modified = True
            print("removing self loops")
            input_g.remove_edges_from(delete_edges)
        
        # 3) Simplify dead ends
        delete_candidates: dict[any,set] = {}
        # first pass to select candidates
        for node_id in input_g.nodes():
            incident_edges = input_g[node_id]
            if len(incident_edges) <= 1:
                    delete_candidates[node_id] = set()
        
        # second pass to count edges arriving to the candidates
        for node_id in input_g.nodes():
            # Exiting from nodes
            incident_edges = input_g[node_id]
            for v_edge in incident_edges:
                if v_edge in delete_candidates:
                    delete_candidates[v_edge].add(node_id)
                if node_id in delete_candidates:
                    delete_candidates[node_id].add(v_edge)
        
        # If the node is only accessed once, remove it
        delete_nodes = []
        for node_id, related_nodes in delete_candidates.items():
            if len(related_nodes) <= 1:
                delete_nodes.append(node_id)

        if len(delete_nodes) > 0:
            is_modified = True
            print("removing dead ends")
            for delete_node in delete_nodes:
                incident_edges = input_g[delete_node]
                for v_edge in incident_edges:
                    input_g._node[v_edge]["aggr_node_number"] += input_g._node[delete_node]["aggr_node_number"]
                input_g.remove_node(delete_node)

        # 4) Simplify gridiron structures
        # candidates = []
        # candidates = set()
        # for node_id in input_g.nodes():
        #     if grid_candidate(node): candidates.add(node_id)

        # print(f"Candidates: {len(candidates)}")
        # print("finished processing initial candidates")
        # # add the I cases to the candidates
        # # for u,v in input_g.edges():
        # #     print(u,v)
        # #     incident_edges_u = list(input_g[u])
        # #     incident_edges_v = list(input_g[v])
        # #     # if all incident nodes of the edge are in the gridiron, add it as well
        # #     if incident_edges_u in candidates and incident_edges_v in candidates:
        # #         candidates.add(u)
        # #         candidates.add(v)
        
        # # print("Finished 'I' special cases")
        # print("removing grid")
        # # Aggregate clusters of grids individually to propagate and distribute nodes
        # while len(candidates) > 0:
        #     candidate = candidates.pop()

        #     remove_candidates = [candidate]
        #     gridiron_entrances = []
        #     aggregated = 0
        #     visit = [candidate]
        #     visited = set()
        #     # recursively visit the individual grid to extract the aggregated value
        #     while True:
        #         if len(visit) == 0: break # end condition
        #         current_node = visit.pop()
        #         if current_node in visited: continue # already visited node
        #         visited.add(current_node)
        #         remove_candidates.append(current_node)
        #         aggregated += input_g._node[current_node]["aggr_node_number"]
        #         current_edges = list(input_g[current_node])
        #         for edg in current_edges:
        #             if edg in candidates and edg not in visited:
        #                 visit.append(edg)
        #             elif edg not in candidates and edg not in visited:
        #                 visited.add(edg)
        #                 gridiron_entrances.append(edg)
            
        #     # obtained the data from a single grid element recursively.
        #     if len(gridiron_entrances) > 0:
        #         distributed_node_agg = aggregated / len(gridiron_entrances)

        #     # distribute node aggregate value to gridiron entrances
        #     for entrance in gridiron_entrances:
        #         input_g._node[entrance]["aggr_node_number"] += distributed_node_agg

        #     # remove the nodes from candidates and the graph
        #     for to_remove in remove_candidates:
        #         if to_remove in candidates:
        #             candidates.remove(to_remove)
        #         if input_g.has_node(to_remove):
        #             input_g.remove_node(to_remove)
        #             is_modified = True

        # 5) Remove interstitial nodes  
        #input_copy = input_g.copy()
        nodes_to_check = list(input_g.nodes())

        # for each endpoint node, look at each of its successor nodes
        for node_id in nodes_to_check:
            if not is_insterstitial(node_id, input_g):
                continue

            neighbors = list(input_g.neighbors(node_id))

            if len(neighbors) != 2:
                continue  # Just a sanity check

            #print(neighbors, node_id)
            n1, n2 = neighbors

            if input_g.has_edge(node_id, n1):
                u = n1
                v = n2
            else: 
                u = n2
                v = n1


            # Get edge geometries
            keys = list(input_g.get_edge_data(node_id, u).keys())
            if len(keys) > 1:
                continue
            
            edge_1 = list(input_g.get_edge_data(node_id, u).values())[0]
            edge_2 = list(input_g.get_edge_data(v, node_id).values())[0]
            
            # Combine geometries
            merged_geom = linemerge(MultiLineString([edge_1['geometry'], edge_2['geometry']]))
            combined_length = edge_1["length"] + edge_2["length"]

            # Remove node and its edges
            removed_node_count = input_g._node[node_id]["aggr_node_number"]
            input_g.remove_node(node_id)

            # update neighbours node counts - sum 0.5 nodes to each node
            input_g._node[u]["aggr_node_number"] += (removed_node_count / 2)
            input_g._node[v]["aggr_node_number"] += (removed_node_count / 2)

            # Add new edge if not already present
            if input_g.has_edge(u, v):
                continue
            
            new_edge_data = {}
            for attribute in edge_1.keys():
                if attribute == "geometry":
                    new_edge_data["geometry"] = merged_geom
                elif attribute == "length":
                    new_edge_data["length"] = combined_length
                else:
                    if attribute not in edge_1:
                        edge_1[attribute] = None
                    if attribute not in edge_2:
                        edge_2[attribute] = None

                    if edge_1[attribute] == edge_2[attribute]:
                        new_edge_data[attribute] = edge_1[attribute]
                    elif type(edge_1[attribute]) == list:
                        new_edge_data[attribute] = edge_1[attribute].append(edge_2[attribute])
                    elif type(edge_2[attribute]) == list:
                        new_edge_data[attribute] = edge_2[attribute].append(edge_1[attribute])
                    else:
                        new_edge_data[attribute] = edge_1[attribute]
                        
                #TODO manage other attributes
                
            input_g.add_edge(u, v, **new_edge_data)

        print(f"Ending Nodes: {input_g.number_of_nodes()}")
        print(f"Ending Edges: {input_g.number_of_edges()}")

    # Return the simplified graph
    print(f"Final: {input_g}")
    input_g.graph = graph_metadata
    return input_g