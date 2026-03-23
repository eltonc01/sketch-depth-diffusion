import numpy as np

class Edge:
    '''
    Edge is unique by the edge's hash
    Each edge should have two faces
    '''
    
    def __init__(self, edge, faces=[], orientations=[], dedge=None, index=None, DiscretizedEdge=None, dedge3d=None):
        self.edge = edge
        self.edges = [edge]
        self.faces = faces
        self.orientations = orientations
        self.dedge = dedge
        self.dedge3d = dedge3d
        self.index = index # index among all edges in TopoMapper, for construct faces
        self.DiscretizedEdge = DiscretizedEdge
        
    def add_face(self, face, orientation):
        self.faces.append(face)
        self.orientations.append(orientation)
        assert len(self.faces) <= 2, "Too many faces for one edge"
        
    def __hash__(self):
        return hash(self.edge)
    
    def __eq__(self, other):
        return isinstance(other, Edge) and hash(self) == hash(other)
    
    def same_orientation(self, other):
        dist1 = np.sum(abs(np.array(self.dedge[-1]) - np.array(other.dedge[0])))
        dist2 = np.sum(abs(np.array(other.dedge[-1]) - np.array(self.dedge[0])))
        return dist1 < dist2
        
    def merge(self, other_edge, topo_mapper):
        """
        Merges another edge into this one, updating geometry and face references.
        """
        # 1. Merge the geometric data (your 'same_orientation' logic is good)
        if self.same_orientation(other_edge):
            self.dedge = self.dedge + other_edge.dedge
            self.dedge3d = self.dedge3d + other_edge.dedge3d
            self.edges = self.edges + other_edge.edges
        else:
            self.dedge = other_edge.dedge + self.dedge
            self.dedge3d = other_edge.dedge3d + self.dedge3d
            self.edges = other_edge.edges + self.edges

        for face in other_edge.faces:
            i = face.keys.index(hash(other_edge.edge))
            del face.edges[i]
            del face.edge_orientations[i]
            del face.keys[i]

        del topo_mapper.all_edges[hash(other_edge)]
        return self