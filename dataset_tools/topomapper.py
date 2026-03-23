from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer, discretize_edge
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopTools import TopTools_SequenceOfShape
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
import numpy as np
from dataset_tools.topology.face import Face
from dataset_tools.topology.edge import Edge
from collections import defaultdict
import random
import noise

from dataset_tools.projection_core import d3_to_d2, project_shapes, project_points

class TopoMapper:
    def __init__(self, shape, args):
        self.shape = shape
        self.all_edges = None
        self.all_faces = None
        self.args = args
        self.tol = self.args.tol
        
        # add outline to shape
        outline_edges = self._find_outline_edges()
        self.full_topo = self._add_outline_edges(outline_edges)
        
        # construct all edge-face mappings
        self._construct_mapping()
        
        # project to 2D; each edge has dedge now
        self._project()
        
        # remove sewn edges
        sewn_edge_keys = self._find_sewn_edges()
        self._remove_sewn_edges(sewn_edge_keys)
        
        # For partial depth: use BFS to select a sequential path of edges to keep perfect (no noise)
        # This matches the logic in dataset.py where 10-90% of edges are revealed via BFS traversal
        self.perfect_edge_indices = set()
        enable_partial = getattr(self.args, 'enable_partial_depth', False)
        if enable_partial:
            # Build adjacency list from topology
            adjacency_list = self._build_adjacency_list_from_topo()
            
            # Get BFS traversal path (same as dataset.py)
            full_path = self._get_all_traversal_paths(adjacency_list)
            
            if full_path:
                # Random percentage between 10% and 90% of the path
                min_step = int((len(full_path) - 1) * 0.1)
                max_step = int((len(full_path) - 1) * 0.9)
                step = random.randint(min_step, max_step)
                
                # Take the first 'step' edges from the BFS path as perfect edges
                # This creates a connected subgraph of perfect edges, just like dataset.py
                self.perfect_edge_indices = set(full_path[:step])

        # Noise application is controlled per-sample by flags on args.
        # For backward compatibility, if flags aren't provided, fall back to old behavior:
        apply_jitter = getattr(self.args, 'apply_jitter', False)
        apply_perlin = getattr(self.args, 'apply_perlin', False)

        if apply_jitter:
            jitter_strength = float(getattr(self.args, 'jitter_strength', 0.0))
            if jitter_strength > 0:
                self._apply_3d_jitter(strength_factor=jitter_strength, perfect_edge_indices=self.perfect_edge_indices)
        if apply_perlin:
            perlin_strength = float(getattr(self.args, 'perlin_strength', 0.0))
            perlin_scale = float(getattr(self.args, 'perlin_scale', 1.0))
            if perlin_strength > 0:
                self._apply_perlin_jitter(strength_factor=perlin_strength, scale=perlin_scale, 
                                         perfect_edge_indices=self.perfect_edge_indices)
    
    def _create_frame(self, normal_vector):
        """Creates an orthonormal basis (a 3D frame) from a single vector."""
        # Normalize the primary vector (u-axis)
        u = normal_vector / np.linalg.norm(normal_vector)
        
        # Find a temporary vector not parallel to u
        temp_vec = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(u, temp_vec)) > 0.99:
            temp_vec = np.array([0.0, 1.0, 0.0])
            
        # Create the perpendicular vectors w and v using cross products
        w = np.cross(u, temp_vec)
        w /= np.linalg.norm(w)
        
        v = np.cross(w, u)
        
        return u, v, w

    def _apply_3d_jitter(self, strength_factor=0.15, precision=6, perfect_edge_indices=None):
        """
        Applies uniform jitter scaled linearly by local minimum edge length.
        
        strength_factor: Fraction of the minimum connected edge length to use as max jitter.
                         E.g., 0.1 = jitter up to 10% of shortest connected edge.
                         Range [0, 0.25] recommended, capped internally at 30% for topology safety.
        
        For partial depth: only apply jitter to vertices where ALL connected edges are non-perfect.
        perfect_edge_indices: set of edge.index values (the indices that appear in edge_buffer)
        """
        if strength_factor == 0:
            return
        
        if perfect_edge_indices is None:
            perfect_edge_indices = set()

        # Build vertex → connected edge indices and minimum edge length mappings
        vertex_to_edge_indices = defaultdict(set)
        vertex_to_min_edge_len = {}
        
        for edge_id, edge in self.all_edges.items():
            if len(edge.dedge3d) > 1:
                points = [np.array(p) for p in edge.dedge3d]
                edge_len = np.linalg.norm(points[-1] - points[0])
                
                start_key = tuple(np.round(points[0], decimals=precision))
                end_key = tuple(np.round(points[-1], decimals=precision))
                
                vertex_to_edge_indices[start_key].add(edge.index)
                vertex_to_edge_indices[end_key].add(edge.index)
                
                # Track minimum edge length per vertex (most constrained)
                if start_key not in vertex_to_min_edge_len:
                    vertex_to_min_edge_len[start_key] = edge_len
                else:
                    vertex_to_min_edge_len[start_key] = min(vertex_to_min_edge_len[start_key], edge_len)
                
                if end_key not in vertex_to_min_edge_len:
                    vertex_to_min_edge_len[end_key] = edge_len
                else:
                    vertex_to_min_edge_len[end_key] = min(vertex_to_min_edge_len[end_key], edge_len)

        # Create the jitter map using linear scaling with min edge length
        # Only jitter vertices where ALL connected edges are non-perfect
        snapped_to_original_map = {}
        jitter_map = {}
        for edge in self.all_edges.values():
            if len(edge.dedge3d) > 0:
                for p in [edge.dedge3d[0], edge.dedge3d[-1]]:
                    p_tuple = tuple(p)
                    snapped_key = tuple(np.round(p, decimals=precision))
                    
                    if snapped_key not in snapped_to_original_map:
                        snapped_to_original_map[snapped_key] = p_tuple
                        
                        # Check if ALL connected edge indices are non-perfect
                        connected_indices = vertex_to_edge_indices[snapped_key]
                        all_edges_non_perfect = all(idx not in perfect_edge_indices for idx in connected_indices)
                        
                        if all_edges_non_perfect:
                            # Get minimum edge length for this vertex
                            min_edge_len = vertex_to_min_edge_len.get(snapped_key, 1.0)
                            
                            # Linear scaling: jitter magnitude = fraction of min edge
                            # Cap at 30% to prevent topology breaks
                            max_jitter = min(strength_factor, 0.3) * min_edge_len
                            
                            # Generate and store the jittered point
                            jitter = np.random.uniform(-max_jitter, max_jitter, 3)
                            jitter_map[p_tuple] = np.array(p_tuple) + jitter
                        else:
                            # Keep vertex perfect (no jitter)
                            jitter_map[p_tuple] = np.array(p_tuple)

        # 3. Transform every edge
        for edge in self.all_edges.values():
            if len(edge.dedge3d) < 2:
                continue

            original_points = [np.array(p) for p in edge.dedge3d]
            v_start_orig, v_end_orig = original_points[0], original_points[-1]
            
            snapped_start_key = tuple(np.round(v_start_orig, decimals=precision))
            representative_start = snapped_to_original_map[snapped_start_key]
            v_start_new = jitter_map[representative_start]

            snapped_end_key = tuple(np.round(v_end_orig, decimals=precision))
            representative_end = snapped_to_original_map[snapped_end_key]
            v_end_new = jitter_map[representative_end]
            
            vec_orig = v_end_orig - v_start_orig
            vec_new = v_end_new - v_start_new
            
            len_orig = np.linalg.norm(vec_orig)
            len_new = np.linalg.norm(vec_new)

            if len_orig < 1e-9:
                edge.dedge3d = [v_start_new for _ in original_points]
                continue

            u_orig, v_orig, w_orig = self._create_frame(vec_orig)
            u_new, v_new, w_new = self._create_frame(vec_new)
            
            scale = len_new / len_orig
            
            new_points = [v_start_new]
            
            for p_orig in original_points[1:-1]:
                disp_vec = p_orig - v_start_orig
                coord_u, coord_v, coord_w = np.dot(disp_vec, u_orig), np.dot(disp_vec, v_orig), np.dot(disp_vec, w_orig)
                disp_vec_new = (scale * coord_u * u_new) + (scale * coord_v * v_new) + (scale * coord_w * w_new)
                p_new = v_start_new + disp_vec_new
                new_points.append(p_new)
                
            new_points.append(v_end_new)
            edge.dedge3d = new_points

    # scale between 0.25 and 5.0
    def _apply_perlin_jitter(self, strength_factor=0.15, scale=1.0, octaves=4, 
                               persistence=0.5, lacunarity=2.0, seed=None, precision=6, perfect_edge_indices=None):
        """
        Applies a smooth, correlated jitter using Perlin noise to simulate an
        unsteady hand. The magnitude is scaled by the square root of the local 
        minimum edge length, providing better balance between small and large features.
        For partial depth: only apply jitter to vertices where ALL connected edges are non-perfect.
        perfect_edge_indices: set of edge.index values (the indices that appear in edge_buffer)
        """
        if strength_factor == 0:
            return
        
        if perfect_edge_indices is None:
            perfect_edge_indices = set()

        # Use a seed for reproducibility
        if seed is None:
            seed = random.randint(0, 1000)

        # Build vertex-to-edge-indices mapping to check connectivity
        vertex_to_edge_indices = defaultdict(set)
        for edge_id, edge in self.all_edges.items():
            if len(edge.dedge3d) > 1:
                points = [np.array(p) for p in edge.dedge3d]
                start_key = tuple(np.round(points[0], decimals=precision))
                end_key = tuple(np.round(points[-1], decimals=precision))
                vertex_to_edge_indices[start_key].add(edge.index)
                vertex_to_edge_indices[end_key].add(edge.index)

        # 1. Map snapped vertices to their connected edge lengths
        snapped_to_edges_map = defaultdict(list)
        for edge in self.all_edges.values():
            if len(edge.dedge3d) > 1:
                points = [np.array(p) for p in edge.dedge3d]
                edge_len = np.linalg.norm(points[-1] - points[0])
                start_key = tuple(np.round(points[0], decimals=precision))
                end_key = tuple(np.round(points[-1], decimals=precision))
                snapped_to_edges_map[start_key].append(edge_len)
                snapped_to_edges_map[end_key].append(edge_len)
        
        # 2. Create the jitter map using Perlin noise
        # Only jitter vertices where ALL connected edges are non-perfect
        snapped_to_original_map = {}
        jitter_map = {}
        
        # Use large offsets to sample different parts of the noise space for each axis
        offset_x = random.uniform(100, 200)
        offset_y = random.uniform(100, 200)
        offset_z = random.uniform(100, 200)

        for edge in self.all_edges.values():
            if len(edge.dedge3d) > 0:
                for p in [edge.dedge3d[0], edge.dedge3d[-1]]:
                    p_tuple = tuple(p)
                    snapped_key = tuple(np.round(p, decimals=precision))
                    
                    if snapped_key not in snapped_to_original_map:
                        snapped_to_original_map[snapped_key] = p_tuple
                        
                        # Check if ALL connected edge indices are non-perfect
                        connected_indices = vertex_to_edge_indices[snapped_key]
                        all_edges_non_perfect = all(idx not in perfect_edge_indices for idx in connected_indices)
                        
                        if all_edges_non_perfect:
                            # --- Perlin Noise Logic ---
                            # Calculate local jitter strength using square root scaling
                            # This compresses the range: small features get relatively more, large get relatively less
                            min_edge_len = np.mean(snapped_to_edges_map[snapped_key]) if snapped_to_edges_map[snapped_key] else 1.0
                            local_strength = np.sqrt(min_edge_len) * strength_factor
                            local_strength = min(local_strength, min(snapped_to_edges_map[snapped_key]) * 0.4)

                            # Use vertex coords to sample the noise field
                            x, y, z = p_tuple
                            nx, ny, nz = x / scale, y / scale, z / scale

                            # Sample 3D noise for each component of the jitter vector
                            jitter_vec = np.array([
                                noise.pnoise3(nx + offset_x, ny, nz, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed),
                                noise.pnoise3(nx, ny + offset_y, nz, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed),
                                noise.pnoise3(nx, ny, nz + offset_z, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed)
                            ])
                            
                            # Normalize the noise vector and scale it by the local strength
                            jitter_vec = jitter_vec / np.linalg.norm(jitter_vec) * local_strength
                            
                            jitter_map[p_tuple] = np.array(p_tuple) + jitter_vec
                        else:
                            # Keep vertex perfect (no jitter)
                            jitter_map[p_tuple] = np.array(p_tuple)
        
        # 3. Transform every edge (this part remains the same)
        for edge in self.all_edges.values():
            if len(edge.dedge3d) < 2:
                continue
            # ... (The rest of the transformation logic is identical to the previous version)
            original_points = [np.array(p) for p in edge.dedge3d]
            v_start_orig, v_end_orig = original_points[0], original_points[-1]
            
            snapped_start_key = tuple(np.round(v_start_orig, decimals=precision))
            representative_start = snapped_to_original_map[snapped_start_key]
            v_start_new = jitter_map[representative_start]

            snapped_end_key = tuple(np.round(v_end_orig, decimals=precision))
            representative_end = snapped_to_original_map[snapped_end_key]
            v_end_new = jitter_map[representative_end]
            
            vec_orig = v_end_orig - v_start_orig
            vec_new = v_end_new - v_start_new
            
            len_orig = np.linalg.norm(vec_orig)
            len_new = np.linalg.norm(vec_new)

            if len_orig < 1e-9:
                edge.dedge3d = [v_start_new for _ in original_points]
                continue

            u_orig, v_orig, w_orig = self._create_frame(vec_orig)
            u_new, v_new, w_new = self._create_frame(vec_new)
            
            scale_warp = len_new / len_orig # Renamed to avoid confusion with noise scale
            
            new_points = [v_start_new]
            
            for p_orig in original_points[1:-1]:
                disp_vec = p_orig - v_start_orig
                coord_u, coord_v, coord_w = np.dot(disp_vec, u_orig), np.dot(disp_vec, v_orig), np.dot(disp_vec, w_orig)
                disp_vec_new = (scale_warp * coord_u * u_new) + (scale_warp * coord_v * v_new) + (scale_warp * coord_w * w_new)
                p_new = v_start_new + disp_vec_new
                new_points.append(p_new)
                
            new_points.append(v_end_new)
            edge.dedge3d = new_points

    def _find_outline_edges(self):
        hlr_shapes = project_shapes(self.shape, self.args)
        outline_compound = hlr_shapes.OutLineVCompound3d()
        if outline_compound:
            return list(TopologyExplorer(outline_compound).edges())
        return []

    def _num_edges(self, splitshape):
        probing_shape = splitshape.Shape()
        split = BRepFeat_SplitShape(probing_shape)
        return split, len(list(TopologyExplorer(probing_shape).edges()))

    def _add_edge(self, split, edge, num_edge):
        toptool_seq_shape = TopTools_SequenceOfShape()
        toptool_seq_shape.Append(edge)
        add_success = split.Add(toptool_seq_shape)
        split, curr_num_edge = self._num_edges(split)
        add_success = add_success and (curr_num_edge > num_edge)
        return split, curr_num_edge, add_success
        
    def _add_outline_edges(self, outline_edges):
        if len(outline_edges) == 0:
            return TopologyExplorer(self.shape)
        split_edge_num = 0
        while True:
            # repeated split edge until number of edges converge
            split = BRepFeat_SplitShape(self.shape)
            split, num_edge = self._num_edges(split)
            for edge in outline_edges: 
                probing_shape = split.Shape()
                backup_split, split = BRepFeat_SplitShape(probing_shape), BRepFeat_SplitShape(probing_shape)
                split, curr_num_edge, add_success = self._add_edge(split, edge, num_edge)
                if not add_success:
                    # Increase outline tolerance when add fails
                    # fixed tolerance, may need update
                    tol = ShapeFix_ShapeTolerance()
                    tol.SetTolerance(edge, 1)
                    split, curr_num_edge, add_success = self._add_edge(backup_split, edge, num_edge)
                    if not add_success:
                        raise Exception("Fail to add splitting outline")
            if split_edge_num == curr_num_edge:
                break
            split_edge_num = curr_num_edge

        split_shape = split.Shape()
        return TopologyExplorer(split_shape)
        
    def _construct_mapping(self):
        '''
        Construct edge-to-face mapping from wireframe.
        '''
        all_edges = {}
        all_faces = {}

        for face in self.full_topo.faces():
            new_face = Face(face, self)
            all_faces[hash(face)] = new_face

            sharp_edges_wires = list(self.full_topo.wires_from_face(face))
            sharp_edges_3d = []
            for wire in sharp_edges_wires:
                sharp_edges_3d += list(WireExplorer(wire).ordered_edges())

            for edge in sharp_edges_3d:
                edge_id = hash(edge) # same edge has same hash
                
                # create edge
                if edge_id in all_edges:
                    new_edge = all_edges[edge_id]
                    new_edge.add_face(new_face, edge.Orientation())
                else:
                    edge_index = len(all_edges) + 1
                    new_edge = Edge(edge, faces=[new_face], orientations=[edge.Orientation()], index=edge_index)
                    all_edges[edge_id] = new_edge
                
                # add edge to face
                new_face.add_edge(new_edge, edge.Orientation())
                
        self.all_faces = all_faces
        self.all_edges = all_edges
        
    def _find_sewn_edges(self):
        '''
        Any edge that occur in any face twice is sewn edge.
        '''
        all_sewn_edge_keys = []
        topo = TopologyExplorer(self.shape)
        for face in topo.faces():
            edge_keys = []
            
            sharp_edges_wires = list(topo.wires_from_face(face))
            sharp_edges_3d = []
            for wire in sharp_edges_wires:
                sharp_edges_3d += list(WireExplorer(wire).ordered_edges())

            for edge in sharp_edges_3d:
                edge_id = hash(edge) # same edge has same hash
                
                # if edge is used twice in a face, it's a sewn edge
                if edge_id in edge_keys:
                    all_sewn_edge_keys.append(edge_id)
                else:
                    edge_keys.append(edge_id)
                    
        return all_sewn_edge_keys

    def _remove_sewn_edges(self, sewn_edge_keys):
        candidate_edges = set()
        for key in sewn_edge_keys:
            # if key in self.all_edges:
            sewn_edge = self.all_edges[key]
            # else:
            #     # sewn edge not found after adding outline

            faces = sewn_edge.faces
            # roll edge sequence
            for face in faces:
                ind = face.keys.index(key)
                face.roll(ind)
            result_face = faces[0]
            for face in faces[1:]:
                pairs = result_face.merge(face)
                if pairs:
                    for pair in pairs:
                        candidate_edges.add(tuple(sorted(pair)))
            
        # merge candidate edges
        for key1, key2 in candidate_edges:
            # check if there's a 4th edge connected to this vertex
            d1, d2 = np.array(self.all_edges[key1].dedge), np.array(self.all_edges[key2].dedge)
            dist = lambda t: np.sum((t[0]-t[1])**2)
            p1, p2 = min([(d1[0], d2[0]), (d1[-1], d2[0]), (d1[0], d2[-1]), (d1[-1], d2[-1])], key=dist)
            vertex = (p1+p2) / 2

            skip = False
            for key in self.all_edges:
                if key == key1 or key == key2 or key in sewn_edge_keys:
                    continue
                e = self.all_edges[key]
                if dist((vertex, e.dedge[0])) < self.tol or dist((vertex, e.dedge[-1])) < self.tol:
                    skip = True
                    break
            
            if not skip:
                self.all_edges[key1].merge(self.all_edges[key2], self)
        
    def _project(self):
        for edge in list(self.all_edges.values()):
            sharp_dedge = discretize_edge(edge.edge, self.args.tol)
            edge.dedge3d = project_points(sharp_dedge, self.args)
            edge.dedge = d3_to_d2(edge.dedge3d)

    def _bfs_traversal(self, adj, start_node, visited):
        """
        Performs a BFS traversal for a single component.
        This is the same implementation as in dataset.py to ensure consistency.
        """
        from collections import deque
        q = deque([start_node])
        visited.add(start_node)
        path = []
        while q:
            node = q.popleft()
            path.append(node)

            neighbors = list(adj.get(node, []))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        return path

    def _get_all_traversal_paths(self, adj):
        """
        Finds all traversal paths, correctly handling multiple disconnected components.
        Returns a single, randomized path that covers all edges.
        This is the same implementation as in dataset.py to ensure consistency.
        """
        all_component_paths = []
        nodes_in_a_path = set()
        all_nodes = list(adj.keys())
        
        # Ensure all nodes are in the adjacency list for iteration
        for node_list in adj.values():
            for node in node_list:
                if node not in all_nodes:
                    all_nodes.append(node)

        random.shuffle(all_nodes)

        for node in all_nodes:
            if node in nodes_in_a_path:
                continue
            
            # Run BFS to find the complete path for the current component
            new_path = self._bfs_traversal(adj, node, nodes_in_a_path)
            all_component_paths.append(new_path)
        
        # --- Data Augmentation: Randomize the order of components ---
        random.shuffle(all_component_paths)
        
        # Concatenate the component paths into a single, full traversal path
        full_path = [edge for component in all_component_paths for edge in component]
        
        return full_path

    def _build_adjacency_list_from_topo(self):
        """
        Builds a true adjacency list by finding which edges share vertices
        in the 3D topology.
        Returns a dict where keys are edge indices and values are lists of connected edge indices.
        """
        vertex_to_edge_map = defaultdict(list)
        
        # First, map every vertex to the edges it's a part of
        for edge in self.all_edges.values():
            for vertex in TopologyExplorer(edge.edge).vertices():
                # Use the vertex hash as a unique key
                vertex_to_edge_map[hash(vertex)].append(edge.index)

        # Now, build the edge-to-edge adjacency list
        adj = defaultdict(set)
        for vertex, connected_edges in vertex_to_edge_map.items():
            # If a vertex is shared by two or more edges, they are connected
            if len(connected_edges) > 1:
                for i in range(len(connected_edges)):
                    for j in range(i + 1, len(connected_edges)):
                        edge1_idx = connected_edges[i]
                        edge2_idx = connected_edges[j]
                        adj[edge1_idx].add(edge2_idx)
                        adj[edge2_idx].add(edge1_idx)
        
        # Convert sets to lists for consistent ordering
        return {k: list(v) for k, v in adj.items()}


    def _rasterize_edges(self, xlim, ylim, W, H, z_far, line_diameter=0.01, render_as_tube=True):
        all_edges = list(self.all_edges.values())
        
        # depth_map = np.full((H, W), z_far, dtype=np.float32)
        # edge_map = np.full((H, W), 0, dtype=np.uint16)
        raster_map = defaultdict(list)

        x_min, x_max = xlim
        y_min, y_max = ylim
        
        x_res = (x_max - x_min) / W
        y_res = (y_max - y_min) / H
        inv_x_res = 1.0 / x_res if x_res != 0 else 0
        inv_y_res = 1.0 / y_res if y_res != 0 else 0
        
        for edge in all_edges:
            edge_index = edge.index
            pts_3d = edge.dedge3d
            pixel_max = defaultdict(list)

            for p0, p1 in zip(pts_3d, pts_3d[1:]):
                (x0, y0, z0), (x1, y1, z1) = p0, p1
                u0, v0 = (x0 - x_min) * inv_x_res, (y0 - y_min) * inv_y_res
                u1, v1 = (x1 - x_min) * inv_x_res, (y1 - y_min) * inv_y_res
                
                steps = int(max(abs(u1 - u0), abs(v1 - v0))) + 1

                for i in range(steps + 1):
                    t = i / steps
                    ui, vi = u0 + t * (u1 - u0), v0 + t * (v1 - v0)
                    zi = z0 + t * (z1 - z0)

                    if render_as_tube:
                        line_radius = line_diameter / 2.0
                        pixel_radius_u = line_radius * inv_x_res
                        pixel_radius_v = line_radius * inv_y_res
                        half_width_u = int(np.ceil(pixel_radius_u))
                        half_width_v = int(np.ceil(pixel_radius_v))
                        center_u_pix, center_v_pix = int(round(ui)), int(round(vi))

                        for dv_pix in range(-half_width_v, half_width_v + 1):
                            for du_pix in range(-half_width_u, half_width_u + 1):
                                u_pix, v_pix = center_u_pix + du_pix, center_v_pix + dv_pix

                                if not (0 <= u_pix < W and 0 <= v_pix < H):
                                    continue
                                
                                dist_from_center_u = (u_pix - ui) * x_res
                                dist_from_center_v = (v_pix - vi) * y_res
                                dist_sq = dist_from_center_u**2 + dist_from_center_v**2
                                
                                if dist_sq > line_radius**2:
                                    continue
                                
                                depth_offset = np.sqrt(line_radius**2 - dist_sq)
                                z_new = zi - depth_offset

                                pixel_max[(v_pix, u_pix)].append((z_new))

                                # if z_new < depth_map[v_pix, u_pix]:
                                    # depth_map[v_pix, u_pix] = z_new
                                    # edge_map[v_pix, u_pix] = edge_index

                    else:
                        u_pix, v_pix = int(round(ui)), int(round(vi))

                        if not (0 <= u_pix < W and 0 <= v_pix < H):
                            continue

                        pixel_max[(v_pix, u_pix)].append((zi))

                        # if zi < depth_map[v_pix, u_pix]:
                            # depth_map[v_pix, u_pix] = zi
                            # edge_map[v_pix, u_pix] = edge_index

            for v_pix, u_pix in list(pixel_max.keys()):
                raster_map[(v_pix, u_pix)].append((min(pixel_max[(v_pix, u_pix)]), edge_index))

        depth_map = np.full((H, W), z_far, dtype=np.float16)
        edge_map = np.full((H, W), 0, dtype=np.uint16)
        depth_intersection = {}
        edge_intersection = {}

        for (y, x), depths in raster_map.items():
            sorted_depths = sorted(depths, key=lambda x: x[0])
            
            depth_map[y, x] = sorted_depths[0][0]
            edge_map[y, x] = sorted_depths[0][1]
            
            if len(sorted_depths) > 1:
                depth_intersection[(y, x)] = [x[0] for x in sorted_depths[1:]]
                edge_intersection[(y, x)] = [x[1] for x in sorted_depths[1:]]
                            
        mask = (depth_map < z_far - 1e-8)
        if mask.any():
            median = np.median(depth_map[mask])
            depth_map[mask] = depth_map[mask] + (z_far / 2) - median

        for coords in list(edge_intersection.keys()):
            depth_intersection[coords] = np.array([x + (z_far / 2) - median for x in depth_intersection[coords]], dtype=np.float16)
            edge_intersection[coords] = np.array(edge_intersection[coords], dtype=np.uint16)

        coords = np.array(list(edge_intersection.keys()), dtype=np.uint16)
        depth_fragments = np.array(list(depth_intersection.values()), dtype=object)
        edge_fragments = np.array(list(edge_intersection.values()), dtype=object)

        return depth_map, depth_fragments, edge_map, edge_fragments, coords
        # return np.flipud(depth_map), np.flipud(edge_map)
        # return depth_map, None, None, None, None

    def generate_pairs(self, save_path_zip, W, H, xlim=(-1, 1), ylim=(-1, 1), z_far=10, line_diameter=0.02, render_as_tube=True):
        depth_buffer, depth_fragments, edge_buffer, edge_fragments, coords = self._rasterize_edges(xlim, ylim, W, H, z_far, 
                                                                                  line_diameter=line_diameter,
                                                                                  render_as_tube=render_as_tube)

        adjacency_list = self._build_adjacency_list_from_topo()

        if np.all(depth_buffer == z_far):
            raise Exception("Camera angle results in empty image.")

        np.savez_compressed(
            save_path_zip,
            depth_buffer=depth_buffer,
            edge_buffer=edge_buffer,
            depth_fragments=depth_fragments,
            edge_fragments=edge_fragments,
            coords=coords,
            adjacency_list=np.array(list(adjacency_list.items()), dtype=object),
            perfect_edges=np.array(list(self.perfect_edge_indices), dtype=np.int64) if self.perfect_edge_indices else np.array([], dtype=np.int64)
        )
    
    def generate_pairs_to_memory(self, W, H, xlim=(-1, 1), ylim=(-1, 1), z_far=10, line_diameter=0.02, render_as_tube=True):
        """Generate rendering data and return as dictionary instead of saving to disk."""
        depth_buffer, depth_fragments, edge_buffer, edge_fragments, coords = self._rasterize_edges(
            xlim, ylim, W, H, z_far, 
            line_diameter=line_diameter,
            render_as_tube=render_as_tube
        )
        
        adjacency_list = self._build_adjacency_list_from_topo()
        
        if np.all(depth_buffer == z_far):
            raise Exception("Camera angle results in empty image.")
        
        return {
            'depth_buffer': depth_buffer,
            'edge_buffer': edge_buffer,
            'depth_fragments': depth_fragments,
            'edge_fragments': edge_fragments,
            'coords': coords,
            'adjacency_list': adjacency_list,
            'perfect_edges': list(self.perfect_edge_indices) if self.perfect_edge_indices else []
        }
