from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def get_boundingbox(shapes, tol=1e-6):
    """Compute center and extent from a list of OCC shapes."""
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    for shape in shapes:
        brepbndlib.Add(shape, bbox, False)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
    extent = abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)
    return center, extent
