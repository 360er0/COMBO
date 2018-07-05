# https://github.com/mrshu/neural-dependency-parser/blob/master/mst_Tim_Dozat.py

import numpy as np


def find_cycles(edges):
    vertices = np.arange(len(edges))
    indices = np.zeros_like(vertices) - 1
    lowlinks = np.zeros_like(vertices) - 1
    stack = []
    onstack = np.zeros_like(vertices, dtype=np.bool)
    current_index = 0
    cycles = []
    
    def strong_connect(vertex, current_index):
        indices[vertex] = current_index
        lowlinks[vertex] = current_index
        stack.append(vertex)
        current_index += 1
        onstack[vertex] = True
        
        for vertex_ in np.where(edges == vertex)[0]:
            if indices[vertex_] == -1:
                current_index = strong_connect(vertex_, current_index)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
            elif onstack[vertex_]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])
        
        if lowlinks[vertex] == indices[vertex]:
            cycle = []
            vertex_ = -1
            while vertex_ != vertex:
                vertex_ = stack.pop()
                onstack[vertex_] = False
                cycle.append(vertex_)
            if len(cycle) > 1:
                cycles.append(np.array(cycle))
        return current_index
    
    for vertex in vertices:
        if indices[vertex] == -1:
            current_index = strong_connect(vertex, current_index)
    return cycles


def find_roots(edges):    
    return np.where(edges[1:] == 0)[0] + 1


def score_edges(probs, edges):
    return np.sum(probs[np.arange(1, len(probs)), edges[1:]])


def chu_liu_edmonds(probs):
    vertices = np.arange(len(probs))
    edges = np.argmax(probs, axis=1)
    cycles = find_cycles(edges)
    if cycles:
        cycle_vertices = cycles.pop()
        non_cycle_vertices = np.delete(vertices, cycle_vertices)
        cycle_edges = edges[cycle_vertices]
        non_cycle_probs = np.array(probs[non_cycle_vertices,:][:,non_cycle_vertices])
        non_cycle_probs = np.pad(non_cycle_probs, [[0,1], [0,1]], 'constant')
        backoff_cycle_probs = probs[cycle_vertices][:,non_cycle_vertices] / probs[cycle_vertices,cycle_edges][:,None]
        non_cycle_probs[-1,:-1] = np.max(backoff_cycle_probs, axis=0)
        non_cycle_probs[:-1,-1] = np.max(probs[non_cycle_vertices][:,cycle_vertices], axis=1)
        non_cycle_edges = chu_liu_edmonds(non_cycle_probs)
        non_cycle_root, non_cycle_edges = non_cycle_edges[-1], non_cycle_edges[:-1]
        source_vertex = non_cycle_vertices[non_cycle_root]
        cycle_root = np.argmax(backoff_cycle_probs[:,non_cycle_root])
        target_vertex = cycle_vertices[cycle_root]
        edges[target_vertex] = source_vertex
        mask = np.where(non_cycle_edges < len(non_cycle_probs)-1)
        edges[non_cycle_vertices[mask]] = non_cycle_vertices[non_cycle_edges[mask]]
        mask = np.where(non_cycle_edges == len(non_cycle_probs)-1)
        stuff = np.argmax(probs[non_cycle_vertices][:,cycle_vertices], axis=1)
        stuff2 = cycle_vertices[stuff]
        stuff3 = non_cycle_vertices[mask]
        edges[stuff3] = stuff2[mask]
    return edges


def mst(probs):
    probs *= 1 - np.eye(len(probs)).astype(np.float32)
    probs[0] = 0
    probs[0, 0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)
    
    edges = chu_liu_edmonds(probs)
    roots = find_roots(edges)
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        for root_idx in roots:
            edges_ = edges.copy()
            for i in range(len(edges)):
                if i != 0 and edges[i] == 0 and i != root_idx:
                    edges_[i] = root_idx

            score = score_edges(probs, edges_)
            if score > best_score:
                best_edges = edges_
                best_score = score

    return best_edges

