from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from diff_drive_rl.tasks.task4.task4_config import Task4Config
from diff_drive_rl.tasks.task4.task4_world import Task4WorldConfig, get_lidar_cfg, spawn_world_assets


def make_diff_drive_task4_scene_cfg(task_cfg: Task4Config | None = None):
    """Build the Task4 InteractiveSceneCfg class.

    Important:
        This function only defines the scene config. It does not launch
        Isaac Sim, does not instantiate InteractiveScene, and does not start
        training. AppLauncher must be started by tests / train scripts before
        importing code paths that require USD / PhysX runtime.

    Design:
        - Four Jetbot articulations are registered as robot_0 ... robot_3.
        - Four RayCaster sensors are kept for compatibility / optional debug.
        - The production actor observation uses analytic LiDAR from
          Task4WorldManager, not RayCaster output.
        - Static world assets are spawned by spawn_task4_world_assets().
    """

    if task_cfg is None:
        task_cfg = Task4Config()

    task_cfg.validate()

    shared_jetbot_actuator = ImplicitActuatorCfg(
        joint_names_expr=[".*wheel_joint"],
        effort_limit_sim=400.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=10.0,
    )

    shared_jetbot_spawn = sim_utils.UsdFileCfg(
        usd_path=str(task_cfg.jetbot_usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    )

    spawn_height = float(task_cfg.spawn_height)
    world_cfg = task_cfg.world_cfg

    @configclass
    class DiffDriveTask4SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(task_cfg.num_envs)
        env_spacing: float = float(task_cfg.env_spacing)

        robot_0: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot_0",
            spawn=shared_jetbot_spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_height)),
            actuators={"drive": shared_jetbot_actuator},
        )

        robot_1: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1",
            spawn=shared_jetbot_spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_height)),
            actuators={"drive": shared_jetbot_actuator},
        )

        robot_2: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2",
            spawn=shared_jetbot_spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_height)),
            actuators={"drive": shared_jetbot_actuator},
        )

        robot_3: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot_3",
            spawn=shared_jetbot_spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_height)),
            actuators={"drive": shared_jetbot_actuator},
        )

        # RayCaster sensors are retained for compatibility and optional
        # visualization/debug. Task4 actor observation uses analytic LiDAR.
        lidar_0 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_0/chassis", world_cfg)
        lidar_1 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_1/chassis", world_cfg)
        lidar_2 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_2/chassis", world_cfg)
        lidar_3 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_3/chassis", world_cfg)

    return DiffDriveTask4SceneCfg


def spawn_task4_world_assets(scene_cfg: InteractiveSceneCfg, task_cfg: Task4Config) -> None:
    """Spawn Task4 static world assets and register RigidObjectCfg entries."""

    task_cfg.validate()
    spawn_world_assets(scene_cfg, task_cfg.world_cfg)


JetbotTask4SceneCfgFactory = make_diff_drive_task4_scene_cfg
DiffDriveTask4SceneCfgFactory = make_diff_drive_task4_scene_cfg
