import networkx as nx
import numpy as np

def greedyForwardPathSearch(spp, result, start, goal, **kwargs):
    # Extract path with a tree walk
    vertices = [start]
    active_edges = []
    unused_edges = spp.Edges()
    max_phi = 0
    max_edge = None
    for edge in unused_edges:
        phi = result.GetSolution(edge.phi())
        if edge.u() == start and phi > max_phi:
            max_phi = phi
            max_edge = edge
    if max_edge is None:
        return None
    active_edges.append(max_edge)
    unused_edges.remove(max_edge)
    vertices.append(max_edge.v())
    
    while active_edges[-1].v() != goal:
        max_phi = 0
        max_edge = None
        for edge in unused_edges:
            phi = result.GetSolution(edge.phi())
            if edge.u() == active_edges[-1].v() and phi > max_phi:
                max_phi = phi
                max_edge = edge
        if max_edge is None:
            return None
        active_edges.append(max_edge)
        unused_edges.remove(max_edge)
        if max_edge.v() in vertices:
            loop_index = vertices.index(max_edge.v())
            active_edges = active_edges[:loop_index]
            vertices = vertices[:loop_index+1]
        else:
            vertices.append(max_edge.v())
    return active_edges

def greedyBackwardPathSearch(spp, result, start, goal, **kwargs):
    # Extract path with a tree walk
    vertices = [goal]
    active_edges = []
    unused_edges = spp.Edges()
    max_phi = 0
    max_edge = None
    for edge in unused_edges:
        phi = result.GetSolution(edge.phi())
        if edge.v() == goal and phi > max_phi:
            max_phi = phi
            max_edge = edge
    if max_edge is None:
        return None
    active_edges.insert(0, max_edge)
    unused_edges.remove(max_edge)
    vertices.insert(0, max_edge.u())

    while active_edges[0].u() != start:
        max_phi = 0
        max_edge = None
        for edge in unused_edges:
            phi = result.GetSolution(edge.phi())
            if edge.v() == active_edges[0].u() and phi > max_phi:
                max_phi = phi
                max_edge = edge
        if max_edge is None:
            return None
        active_edges.insert(0, max_edge)
        unused_edges.remove(max_edge)
        if max_edge.u() in vertices:
            loop_index = vertices.index(max_edge.u())
            active_edges = active_edges[loop_index+1:]
            vertices = vertices[loop_index:]
        else:
            vertices.insert(0, max_edge.u())
    return active_edges

def averageVertexPositionSpp(gcs, result, start, goal, edge_cost_dict=None, flow_min=1e-4, **kwargs):
    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())

    vertex_data = {}
    for v in gcs.Vertices():
        vertex_data[v.id()] = np.zeros(v.set().ambient_dimension() + 1)

    for e in gcs.Edges():
        vertex_data[e.u().id()][:-1] += e.GetSolutionPhiXu(result)
        vertex_data[e.u().id()][-1] += result.GetSolution(e.phi())
        if e.v() == goal:
            vertex_data[goal.id()][:-1] += e.GetSolutionPhiXv(result)
            vertex_data[goal.id()][-1] += result.GetSolution(e.phi())

    for v in gcs.Vertices():
        if vertex_data[v.id()][-1] > flow_min:
            vertex_data[v.id()] = vertex_data[v.id()][:-1] / vertex_data[v.id()][-1]
        else:
            vertex_data[v.id()] = np.zeros(v.set().ambient_dimension())

    for e in gcs.Edges():
        G.add_edge(e.u(), e.v())
        e_cost = 0
        for cost in edge_cost_dict[e.id()]:
            if len(cost.variables()) == e.u().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(vertex_data[e.u().id()])
            elif len(cost.variables()) == e.u().set().ambient_dimension() + e.v().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(np.append(vertex_data[e.u().id()], vertex_data[e.v().id()]))
            else:
                raise Exception("Unclear what variables are used in this cost.")
        G.edges[e.u(), e.v()]['l'] = np.squeeze(e_cost)
        if G.edges[e.u(), e.v()]['l'] < 0:
            G.edges[e.u(), e.v()]['l'] = np.inf

    path = nx.bidirectional_dijkstra(G, start, goal, 'l')[1]

    active_edges = []
    for u, v in zip(path[:-1], path[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                active_edges.append(e)
                break

    return active_edges

def dijkstraRounding(gcs, result, source, target, flow_min=1e-4, **kwargs):
    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())
    G.add_edges_from([(e.u(), e.v()) for e in gcs.Edges()])

    for e in gcs.Edges():
        flow = result.GetSolution(e.phi())
        if flow > flow_min:
            G.edges[e.u(), e.v()]['l'] = e.GetSolutionCost(result) / flow

    path = nx.bidirectional_dijkstra(G, source, target, 'l')[1]

    active_edges = []
    for u, v in zip(path[:-1], path[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                active_edges.append(e)
                break

    return active_edges
