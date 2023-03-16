from pydrake.all import (
    PRMPlannerCreationParameters,
    PRMPlanner,
    PathProcessor,
    HolonomicKinematicPlanningSpace,
    JointLimits,
    BiRRTPlanner,
    SceneGraphCollisionChecker,
    VoxelizedEnvironmentCollisionChecker,
    MbpEnvironmentCollisionChecker,
    BuildPRMsplinedBiRRT,
    RobotDiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    ChangeOmpNumThreadsWrapper,
)

from reproduction.util import FindModelFile, GcsDir
from itertools import combinations
import numpy as np


class PresplinedPRM:
    def __init__(
        self,
        edge_step_size=0.05,
        env_padding=0.01,
        self_padding=0.01,
        propagation_step_size=0.5,
        grid_size=[2.0, 2.0, 2.0],
        grid_resolution=0.02,
        seed=0,
    ):
        # Build the Scene

        builder = RobotDiagramBuilder(time_step=0.0)
        builder.parser().package_map().Add("gcs", GcsDir())

        directives_file = FindModelFile(
            "models/iiwa14_spheres_collision_welded_gripper.yaml"
        )
        directives = LoadModelDirectives(directives_file)
        ProcessModelDirectives(directives, builder.parser())

        iiwa_idx = builder.plant().GetModelInstanceByName("iiwa")
        wsg_idx = builder.plant().GetModelInstanceByName("wsg")
        named_joint_distance_weights = dict()
        named_joint_distance_weights["iiwa_joint_1"] = 1.0
        named_joint_distance_weights["iiwa_joint_2"] = 1.0
        named_joint_distance_weights["iiwa_joint_3"] = 1.0
        named_joint_distance_weights["iiwa_joint_4"] = 1.0
        named_joint_distance_weights["iiwa_joint_5"] = 1.0
        named_joint_distance_weights["iiwa_joint_6"] = 1.0
        named_joint_distance_weights["iiwa_joint_7"] = 1.0

        # named_joint_distance_weights["iiwa_joint_1"] = 2.0
        # named_joint_distance_weights["iiwa_joint_2"] = 2.0
        # named_joint_distance_weights["iiwa_joint_3"] = 2.0
        # named_joint_distance_weights["iiwa_joint_4"] = 2.0
        # named_joint_distance_weights["iiwa_joint_5"] = 1.0
        # named_joint_distance_weights["iiwa_joint_6"] = 1.0
        # named_joint_distance_weights["iiwa_joint_7"] = 1.0

        builder.plant().Finalize()
        joint_limits = JointLimits(builder.plant())
        diagram = builder.Build()

        robot_model_instances = [iiwa_idx, wsg_idx]
        # Build collision checker

        collision_checker = VoxelizedEnvironmentCollisionChecker(
            model=diagram,
            robot_model_instances=robot_model_instances,
            edge_step_size=edge_step_size,
            env_collision_padding=env_padding,
            self_collision_padding=self_padding,
            named_joint_distance_weights=named_joint_distance_weights,
            default_joint_distance_weight=1.0,
        )

        collision_checker.VoxelizeEnvironment(grid_size, grid_resolution)

        # Make the planning space
        self.planning_space = HolonomicKinematicPlanningSpace(
            collision_checker, joint_limits, propagation_step_size, seed
        )
        self.roadmap = None

    def build(self, birrt_param, prm_creation_param, milestone_configurations):
        # Build initial PRM roadmap
        splined_path = []
        birrt_times = []
        for start, goal in combinations(milestone_configurations.values(), 2):
            # BiRRT connect
            result, runtime = BiRRTPlanner.TimedPlan(
                start,
                goal,
                birrt_param,
                self.planning_space,
            )
            birrt_times.append(runtime)
            if result.has_solution():
                splined_path += result.path()

        splined_path = np.vstack(splined_path)
        self.roadmap, prm_runtime = PRMPlanner.TimedBuildRoadmap(
            prm_creation_param, splined_path, self.planning_space
        )

        return prm_runtime, birrt_times

    def plan(self, sequence, query_paramters, post_processing_parameters=None):
        if self.roadmap is None:
            raise Exception("Roadmap not built yet")
        path = [sequence[0]]
        run_time = 0.0
        for start, goal in zip(sequence[:-1], sequence[1:]):
            # path_result, prm_run_time = PRMPlanner.TimedPlanLazyAddingNodes(
            path_result, prm_run_time = PRMPlanner.TimedPlanLazy(
                start, goal, query_paramters, self.planning_space, self.roadmap
            )
            run_time += prm_run_time

            if not path_result.has_solution():
                print(f"Failed between {start} and {goal}")
                return None, run_time

            if post_processing_parameters is not None:
                processed_path, processing_time = PathProcessor.TimedProcessPath(
                    path_result.path(), post_processing_parameters, self.planning_space
                )
                run_time += processing_time
                path += processed_path[1:]
            else:
                path += path_result.path()[1:]

        return np.stack(path).T, run_time
