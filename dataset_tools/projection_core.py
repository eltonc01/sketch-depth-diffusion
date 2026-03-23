import numpy as np

def randnum(low, high):
    return np.random.rand() * (high - low) + low

# generate a random camera
def generate_random_camera_pos(seed=None):
    if seed:
        np.random.seed(seed)
    focus = randnum(3, 5)
    radius = 3.5 # randnum(1.25, 1.5) # distance of camera to origin
    phi = randnum(0, 180) # longitude, elevation of camera
    theta = randnum(0, 360) # latitude, rotation around z-axis
    return focus, pose_spherical(theta, phi, radius)


def pose_spherical(theta, phi, radius):
    def trans_t(t): return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    def rot_phi(phi): return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    def rot_theta(th): return np.array([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(np.deg2rad(phi)) @ c2w
    c2w = rot_theta(np.deg2rad(theta)) @ c2w
    c2w = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ c2w
    return c2w



def project_shapes(shapes, args):
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

    location = args.location
    direction = args.direction
    focus = args.focus

    hlr = HLRBRep_Algo()

    if isinstance(shapes, list):
        for shape in shapes:
            hlr.Add(shape)
    else:
        hlr.Add(shapes)
    ax = gp_Ax2(gp_Pnt(*location), gp_Dir(*direction))

    if args.pose is not None:
        pose = args.pose
        ax = gp_Ax2(gp_Pnt(*pose[:3, -1].tolist()), gp_Dir(*pose[:3, -2].tolist()), gp_Dir(*pose[:3, 0].tolist()))

    if focus == 0:
        projector = HLRAlgo_Projector(ax)
    else:
        projector = HLRAlgo_Projector(ax, focus)

    hlr.Projector(projector)
    hlr.Update()

    hlr_shapes = HLRBRep_HLRToShape(hlr)
    return hlr_shapes
def d3_to_d2(points_3d):
    return [tuple(p[:2]) for p in points_3d]

def project_points(points, args):
    focus = args.focus
    if args.pose is not None:
        # Invert to get world→camera
        cam2world = args.pose            # 4×4
        world2cam = np.linalg.inv(cam2world)
        projected = []
        for p in points:
            # homogeneous world point
            pw = np.array([p[0], p[1], p[2], 1.0], dtype=float)
            pc = world2cam @ pw            # now in camera coords
            x_cam, y_cam, z_cam = pc[:3]

            # if your pose convention ends up with
            # camera looking down –Z, you’ll get z_cam < 0 for
            # visible points.  Flip it to make depth positive:
            z_cam = -z_cam

            if focus and z_cam != 0:
                s = focus / z_cam
                x_cam *= s
                y_cam *= s
            projected.append((x_cam, y_cam, z_cam))
        return projected

