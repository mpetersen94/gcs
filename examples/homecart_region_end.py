import numpy as np
import multiprocessing as mp
import os
import pickle
import time

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, FindResourceOrThrow,
    IrisInConfigurationSpace, IrisOptions,
    LoadModelDirectives, Parser, ProcessModelDirectives
)

from reproduction.util import *

from pydrake.common import configure_logging
configure_logging()

seed_points = {
    'nominal': [1.2, -0.8, 1.2, -2.1, -1.8, -0.1, -1.3, -2.4, -1.1, -1.3, 1.6, 0],
    'left_grasp': [1.65, -1.06, 1.65, -2.13, -1.58, 0.11, -1.3, -2.4, -1.1, -1.3, 1.6, 0.],
    'hand_off': [1.6 , -0.95, 1.65, -2.15, -3.12, 0., -1.6, -2.19, -1.66, -1., 3.12, 0.],
    'right_grasp': [1.2, -0.8, 1.2, -2.1, -1.8, -0.1, -1.75, -1.99, -1.82, -0.9, 1.6, -0.23],
}
X_sugar_start = RigidTransform(RollPitchYaw(-np.pi/2, 0, 0), [0.36, 0.36, 0.11])
X_sugar_left = RigidTransform(RollPitchYaw(0, -np.pi/2, 0), [0.0, 0.035, 0.0])
X_sugar_right = RigidTransform(RollPitchYaw(0, -np.pi/2, np.pi), [0.0, 0.032, 0.0])
X_sugar_end = RigidTransform(RollPitchYaw(np.pi/2, 0, 0), [0.3250775219397953, -0.3818997004276029, 0.1172])

def parse_homecart(plant):
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())
    directives = LoadModelDirectives(
        FindModelFile('models/homecart.yaml'))
    ProcessModelDirectives(directives, plant, parser)

iris_options = IrisOptions()
iris_options.require_sample_point_is_contained = True
iris_options.iteration_limit = 5
iris_options.termination_threshold = -1
iris_options.relative_termination_threshold = 0.02
iris_options.enable_ibex = False
iris_options.configuration_space_margin = 0.005

# for X, name in zip([X_sugar_start, X_sugar_left, X_sugar_right, X_sugar_end], ["start", "left", "right", "end"]):
X = X_sugar_end
name = "end"

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parse_homecart(plant)
Parser(plant).AddModelFromFile(
    FindResourceOrThrow(
        'drake/manipulation/models/ycb/sdf/004_sugar_box.sdf'))
base_frame = plant.world_frame()
if name == "left":
    base_frame = plant.GetFrameByName('grasp_frame', plant.GetModelInstanceByName("gripper_left"))
elif name == "right":
    base_frame = plant.GetFrameByName('grasp_frame', plant.GetModelInstanceByName("gripper_right"))
plant.WeldFrames(base_frame, plant.GetFrameByName('base_link_sugar'), X)
plant.Finalize()
diagram = builder.Build()

def calcRegion(seed):
    start_time = time.time()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, seed)
    try:
        hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)
        print("Seed:", seed, "\tTime:", time.time() - start_time, flush=True)
        return hpoly
    except:
        print("Seed:", seed, "Failed.")
        return None

def generateRegions(seed_points):
    seeds = list(seed_points.values()) if type(seed_points) is dict else seed_points
    regions = []
    loop_time = time.time()
    with mp.Pool(processes = 4) as pool:
        regions = pool.starmap(calcRegion, [[seed] for seed in seeds])

    print("Loop time:", time.time() - loop_time)

    if type(seed_points) is dict:
        return dict(list(zip(seed_points.keys(), regions)))
    
    for key in regions:
        if regions[key] is None:
            regions.pop(key)

    return regions

print("Starting region generation for", name)
regions = generateRegions(seed_points)

with open(os.path.join(GcsDir(), 'data/homecart/iris_regions_' + name + '.reg'), 'wb') as f:
    pickle.dump(regions,f)
