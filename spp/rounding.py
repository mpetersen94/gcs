import networkx as nx
import numpy as np

def MipPathExtraction(gcs, result, start, goal, **kwargs):
    outgoing_edges = {}
    for v in gcs.Vertices():
        outgoing_edges[v.id()] = []

    for e in gcs.Edges():
        outgoing_edges[e.u().id()].append(e)

    active_edges = []
    for edge in outgoing_edges[start.id()]:
        phi = result.GetSolution(edge.phi())
        if phi > 0.5:
            active_edges.append(edge)
            break

    while active_edges[-1].v() != goal:
        for edge in outgoing_edges[active_edges[-1].v().id()]:
            phi = result.GetSolution(edge.phi())
            if phi > 0.5:
                active_edges.append(edge)
                break
    return [active_edges]

def greedyForwardPathSearch(gcs, result, start, goal, **kwargs):
    # Extract path with a tree walk
    vertices = [start]
    active_edges = []
    unused_edges = gcs.Edges()
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
    return [active_edges]

def greedyBackwardPathSearch(gcs, result, start, goal, **kwargs):
    # Extract path with a tree walk
    vertices = [goal]
    active_edges = []
    unused_edges = gcs.Edges()
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
    return [active_edges]

def randomPathSearch(gcs, result, start, goal, seed=None, num_paths=10, flow_min=1e-8, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    paths = []
    for _ in range(num_paths):

        outgoing_edges = {}
        for v in gcs.Vertices():
            outgoing_edges[v.id()] = []

        for e in gcs.Edges():
            outgoing_edges[e.u().id()].append(e)

        # Extract path with a tree walk
        vertices = [start]
        visited_vertices = [start]
        active_edges = []

        while vertices[-1] != goal:
            e_out = outgoing_edges[vertices[-1].id()]
            phi_values = np.empty(len(e_out))
            for ii in range(len(e_out)):
                phi_values[ii] = result.GetSolution(e_out[ii].phi())
            if len(e_out) == 0 or np.sum(phi_values) < flow_min:
                active_edges.pop()
                vertices.pop()
                if len(vertices) == 0:
                    break
                continue

            edge = np.random.choice(e_out, p=phi_values / np.sum(phi_values))
            e_out.remove(edge)
            if edge.v() not in visited_vertices:
                active_edges.append(edge)
                vertices.append(edge.v())
                visited_vertices.append(edge.v())

        if active_edges[-1].v() == goal:
            paths.append(active_edges)

    if len(paths) == 0:
        return None
    return paths

def averageVertexPositionSpp(gcs, result, start, goal, edge_cost_dict=None, flow_min=1e-3, **kwargs):
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
            vertex_data[v.id()] = v.set().ChebyshevCenter()

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
            raise RuntimeError(f"Averaged length of edge {e} is negative. Consider increasing flow_min.")

    path = nx.bidirectional_dijkstra(G, start, goal, 'l')[1]

    active_edges = []
    for u, v in zip(path[:-1], path[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                active_edges.append(e)
                break

    return [active_edges]

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

    return [active_edges]
