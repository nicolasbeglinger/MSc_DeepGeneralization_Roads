import numpy as np
import numbers
from copy import copy

class Road(object):
    def __init__(self, feature_dict):
        self.coordinates = np.array(feature_dict['geometry']['coordinates'])
        self.original_coordinates = np.array(feature_dict['geometry']['coordinates'])

        self.properties = feature_dict['properties']
        self.num_vertices = feature_dict['properties']['num_vertices']

        self.exagg_value = {0: 0, 1: 0}
    
    def __repr__(self):
        return f"Road with {self.properties['num_vertices']} vertices"

    def exaggerate_axis(self, axis, factor):
        if self.exagg_value[axis] != factor:
            self.coordinates = self.original_coordinates * 1
            self.coordinates[:, axis] *= factor
            self.exagg_value[axis] = factor

class Vertex(object):
    def __init__(self, index, road_index, coords, translation_vector, properties: list, connections):
        self.index = index
        self.road_index = road_index
        self.translation_vector = translation_vector

        for i in properties:
            assert isinstance(i, numbers.Number)


        self.coords = copy(coords)
        self.properties = np.array(self.coords + properties + translation_vector)

        self.connections = connections

    def __str__(self):

        lines = []

        lines.append(f"Vertex {self.index}")
        lines.append(f"Coordinates: {[np.round(coord, 2) for coord in self.coords]}")
        lines.append(f"Connections: {self.connections}")

        lines.append(f"Attributes: {self.properties}")
        # lines.append("\n")

        return "\n".join(lines)
    
    def __repr__(self):
        return f"Vertex {self.index}"

class Graph(object):
    def __init__(self, nodes):
        self.nodes = nodes

        self.adj_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        
        for node in self.nodes:
            for connection in node.connections:
                self.adj_matrix[node.index, connection] = 1
        
        self.node_dict = {node.index: {
            "coords": node.coords,
            "transl": node.translation_vector,
            "props": node.properties,
            "x": node.properties[:-1],
            "y": node.translation_vector} for node in self.nodes}

    def __repr__(self):
        return f"Graph: {len(self.nodes)} Nodes"



    















