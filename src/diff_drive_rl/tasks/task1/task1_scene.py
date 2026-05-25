from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from diff_drive_rl.tasks.task1.task1_config import Task1Config


def make_diff_drive_task1_scene_cfg(cfg: Task1Config):
    """Factory for Diff-Drive UGV / Jetbot Task1 scene config.

    This file intentionally contains no AppLauncher and no environment logic.
    Test / train scripts must start Isaac Lab AppLauncher before importing env.
    """

    @configclass
    class DiffDriveTask1SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(
                color=tuple(float(c) for c in cfg.ground_color),
            ),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0),
        )

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

    return DiffDriveTask1SceneCfg


JetbotTask1SceneCfgFactory = make_diff_drive_task1_scene_cfg
DiffDriveTask1SceneCfgFactory = make_diff_drive_task1_scene_cfg
