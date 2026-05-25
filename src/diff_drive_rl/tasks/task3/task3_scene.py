from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from diff_drive_rl.tasks.task3.task3_config import Task3Config
from diff_drive_rl.tasks.task3.task3_world import get_lidar_cfg, spawn_world_assets


def make_diff_drive_task3_scene_cfg(cfg: Task3Config):
    """Factory for Jetbot Task3 Isaac scene.

    This scene contains:
        - Jetbot articulation
        - conservative manually spawned world assets:
            floors, lane walls, speed bumps, U-shaped parking walls
        - RigidObject registration for selected world objects
        - RayCaster sensor config attached to Jetbot chassis

    Important:
        The task3_world layer provides an analytic 2D LiDAR path that is used
        by task3_env.py for robust training. The RayCaster is still registered
        to keep the project interface complete and to support future visual /
        sensor checks.

        This file must be imported after AppLauncher in executable scripts,
        because task3_world.py imports Isaac Lab asset APIs.
    """

    cfg.validate()

    @configclass
    class DiffDriveTask3SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cfg.jetbot_usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.spawn_height)),
            ),
            actuators={
                "drive": ImplicitActuatorCfg(
                    joint_names_expr=[".*wheel_joint"],
                    effort_limit_sim=400.0,
                    velocity_limit_sim=100.0,
                    stiffness=0.0,
                    damping=10.0,
                )
            },
        )

    # Conservative world assets must be spawned before InteractiveScene
    # construction. This function also registers speed_bump_i and parking walls
    # as RigidObjectCfg entries on the scene cfg.
    spawn_world_assets(DiffDriveTask3SceneCfg, cfg.world_cfg)

    # Attach RayCaster to Jetbot chassis. The environment uses analytic LiDAR
    # for training stability, but keeping this sensor registered is useful for
    # compatibility and future checks.
    DiffDriveTask3SceneCfg.lidar = get_lidar_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        cfg=cfg.world_cfg,
    )

    return DiffDriveTask3SceneCfg


JetbotTask3SceneCfgFactory = make_diff_drive_task3_scene_cfg
DiffDriveTask3SceneCfgFactory = make_diff_drive_task3_scene_cfg
