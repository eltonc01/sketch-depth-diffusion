from __future__ import annotations

from typing import Tuple, Dict


def calculate_shape_complexity(shape) -> Tuple[float, int]:
    """Mirror of generate_abc.calculate_shape_complexity.

    Returns:
        (complexity_score, edge_count)

    complexity_score is a weighted sum of curve primitive types.
    """

    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Geom import (
        Geom_Line,
        Geom_Circle,
        Geom_Ellipse,
        Geom_BSplineCurve,
        Geom_BezierCurve,
    )
    from OCC.Extend.TopologyUtils import TopologyExplorer

    complexity_score = 0.0
    edge_count = 0

    costs = {
        "line": 1,
        "circle": 5,
        "ellipse": 8,
        "bspline": 10,
        "bezier": 10,
        "other": 5,
    }

    for edge in TopologyExplorer(shape).edges():
        edge_count += 1
        try:
            curve_result = BRep_Tool.Curve(edge)
            if curve_result is None or curve_result[0] is None:
                continue
            curve_handle = curve_result[0]
        except Exception:
            continue

        try:
            if Geom_Line.DownCast(curve_handle) is not None:
                complexity_score += costs["line"]
            elif Geom_Circle.DownCast(curve_handle) is not None:
                complexity_score += costs["circle"]
            elif Geom_Ellipse.DownCast(curve_handle) is not None:
                complexity_score += costs["ellipse"]
            elif Geom_BSplineCurve.DownCast(curve_handle) is not None:
                complexity_score += costs["bspline"]
            elif Geom_BezierCurve.DownCast(curve_handle) is not None:
                complexity_score += costs["bezier"]
            else:
                complexity_score += costs["other"]
        except Exception:
            complexity_score += costs["other"]

    return float(complexity_score), int(edge_count)


def get_topology_counts(shape) -> Dict[str, int]:
    """Basic topology counts for a shape (faces/edges/vertices)."""

    from OCC.Extend.TopologyUtils import TopologyExplorer

    topo = TopologyExplorer(shape)
    return {
        "topo_num_faces": len(list(topo.faces())),
        "topo_num_edges": len(list(topo.edges())),
        "topo_num_vertices": len(list(topo.vertices())),
    }
