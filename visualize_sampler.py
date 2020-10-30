import argparse
import trimesh
import torch
import numpy as np
from deep_sdf.samplers import NormalPerPoint as Sampler


def get_sphere_transform(translation):
    t = np.eye(4)
    t[:3,3] = translation
    return t

def add_point_to_scene(scene, point, scale=1., color=[128,128,128,50]):
    sphere = trimesh.primitives.Sphere()
    sphere = trimesh.Trimesh(vertices=sphere.vertices*scale, faces=sphere.faces)
    sphere.apply_transform(get_sphere_transform(translation=point))
    sphere.visual.face_colors = color
    scene.add_geometry(sphere)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--mesh',
        help='Base geometry to sample',
        required=True)
    parser.add_argument(
        '--normalize-scene',
        help='Normalize scene with mean 0 and variance 1',
        action='store_true')
    parser.add_argument(
        '--global-sigma',
        help='Global sigma used by sampler',
        type=float,
        default=1)
    parser.add_argument(
        '--local-sigma',
        help='Local sigma used by sampler',
        type=float,
        default=1e-2)
    parser.add_argument(
        '--max-points',
        help='Max points to be shown',
        type=int,
        default=100)
    parser.add_argument(
        '--point-scale',
        help='Scale of the points',
        type=float,
        default=1)

    args = parser.parse_args()

    scene = trimesh.Scene()

    # Load and add mesh to the scene
    mesh = trimesh.load_mesh(args.mesh)
    scene.add_geometry(mesh, node_name='mesh')

    # Sample points and add them to the scene
    vertices = torch.from_numpy(mesh.vertices).float()
    sampler = Sampler(global_sigma=args.global_sigma, local_sigma=args.local_sigma)
    p_global = sampler.get_points_global(args.max_points//2)
    p_local = sampler.get_points_local(vertices, args.max_points//2)
    for idx, (p_g, p_l) in enumerate(zip(p_global, p_local)):
        add_point_to_scene(scene, p_g, scale=args.point_scale, color=[255,0,0,90])
        add_point_to_scene(scene, p_l, scale=args.point_scale, color=[0,255,0,90])

    scene.show()