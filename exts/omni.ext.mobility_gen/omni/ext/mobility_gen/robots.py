# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

# ================= Standard imports =================
import os
from typing import Any, List, Optional, Tuple, Type, Union

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni.usd
from isaacsim.core.api.robots.robot import Robot as _Robot

# ================= Isaac Sim imports =================
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import (
    AckermannController,
)
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot

# ================= Extension imports =================
from omni.ext.mobility_gen.common import Buffer, Module

# Sensores extras
from omni.ext.mobility_gen.sensors import (
    HawkCamera,
    Sensor,
    ZedStereoCamera,
    FisheyeCamera,
    LidarSensor,
)
from omni.ext.mobility_gen.types import Pose2d
from omni.ext.mobility_gen.utils.global_utils import get_stage, get_world
from omni.ext.mobility_gen.utils.prim_utils import (
    prim_rotate_x,
    prim_rotate_y,
    prim_rotate_z,
    prim_translate,
)
from omni.ext.mobility_gen.utils.registry import Registry
from omni.ext.mobility_gen.utils.stage_utils import (
    stage_add_camera,
    stage_add_usd_ref,
    stage_get_prim,
)
from omni.isaac.core.articulations import ArticulationView as _ArticulationView

# USD / PhysX
from pxr import Gf, UsdGeom

# =====================================================
#  BASE CLASSES
# =====================================================


class Robot(Module):
#inicializando os modulos
    physics_dt: float
    z_offset: float

    chase_camera_base_path: str
    chase_camera_x_offset: float
    chase_camera_z_offset: float
    chase_camera_tilt_angle: float

    occupancy_map_radius: float
    occupancy_map_z_min: float
    occupancy_map_z_max: float
    occupancy_map_cell_size: float
    occupancy_map_collision_radius: float

    main_camera_type: Type[Sensor]
    main_camera_base_path: str
    main_camera_rotation: Tuple[float, float, float]
    main_camera_translation: Tuple[float, float, float]

    fisheye_camera_type: Type[Sensor] = Type[Sensor]
    fisheye_camera_base_path: str
    fisheye_camera_rotation: Tuple[float, float, float]
    fisheye_camera_translation: Tuple[float, float, float]

    lidar_sensor_type: Type[Sensor]
    lidar_sensor_base_path: str
    lidar_sensor_rotation: Tuple[float, float, float]
    lidar_sensor_translation: Tuple[float, float, float]
    lidar_file_name: str
    lidar_sensor_attributes: dict

    def __init__(
        self,
        prim_path: str,
        robot: _Robot,
        articulation_view: _ArticulationView,
        main_camera: Sensor,
        fisheye_camera: Sensor = None,
        lidar_sensor: Sensor = None,
    ):
        self.prim_path = prim_path
        self.robot = robot
        self.articulation_view = articulation_view
        self.action = Buffer(np.zeros(2))
        self.position = Buffer()
        self.orientation = Buffer()
        self.joint_positions = Buffer()
        self.joint_velocities = Buffer()
        self.main_camera = main_camera
        self.fisheye_camera = fisheye_camera
        self.lidar_sensor = lidar_sensor

    @classmethod
    def build_main_camera(cls, prim_path):
        camera_path = os.path.join(prim_path, cls.main_camera_base_path)
        XFormPrim(camera_path)
        stage = get_stage()
        main_camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(main_camera_prim, cls.main_camera_rotation[0])
        prim_rotate_y(main_camera_prim, cls.main_camera_rotation[1])
        prim_rotate_z(main_camera_prim, cls.main_camera_rotation[2])
        prim_translate(main_camera_prim, cls.main_camera_translation)

        # enable RGB rendering so recorder captures images
        sensor = cls.main_camera_type.build(prim_path=camera_path)
        try:
            if hasattr(sensor, "cam") and sensor.cam is not None:
                sensor.cam.enable_rgb_rendering()
        except Exception as e:
            print(f"[Robot.build_main_camera] enable_rgb_rendering skipped: {e}")
        return sensor


    @classmethod
    def build_fisheye_camera(cls, prim_path):
        camera_path = os.path.join(prim_path, cls.fisheye_camera_base_path)
        XFormPrim(camera_path)
        stage = get_stage()
        FisheyeCamera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(FisheyeCamera_prim, cls.fisheye_camera_rotation[0])
        prim_rotate_y(FisheyeCamera_prim, cls.fisheye_camera_rotation[1])
        prim_rotate_z(FisheyeCamera_prim, cls.fisheye_camera_rotation[2])
        prim_translate(FisheyeCamera_prim, cls.fisheye_camera_translation)
        # enable RGB rendering so recorder captures images
        sensor = cls.fisheye_camera_type.build(prim_path=camera_path)
        try:
            if hasattr(sensor, "cam") and sensor.cam is not None:
                sensor.cam.enable_rgb_rendering()
        except Exception as e:
            print(f"[Robot.build_fisheye_camera] enable_rgb_rendering skipped: {e}")
        return sensor
    
    @classmethod
    def build_lidar_sensor(cls, prim_path):
        sensor_path = os.path.join(prim_path, cls.lidar_sensor_base_path)
        # Note: Do NOT create XFormPrim here - LidarRtx will create the OmniLidar prim
        # The translation/rotation are passed directly to LidarRtx
        sensor = cls.lidar_sensor_type.build(
            prim_path=sensor_path,
            translation=cls.lidar_sensor_translation,
            config_file_name=cls.lidar_file_name,
        )
        # Atualizar e desenhar Lidar
        if hasattr(cls, 'lidar_sensor') and cls.lidar_sensor is not None:
            cls.lidar_sensor.update_state()
            cls.lidar_sensor.draw_point_cloud()

        return sensor

    def build_chase_camera(self) -> str:
        stage = get_stage()
        camera_path = os.path.join(
            self.prim_path, self.chase_camera_base_path, "chase_camera"
        )
        stage_add_camera(
            stage,
            camera_path,
            focal_length=10,
            horizontal_aperature=30,
            vertical_aperature=30,
        )
        camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(camera_prim, self.chase_camera_tilt_angle)
        prim_rotate_y(camera_prim, 0)
        prim_rotate_z(camera_prim, 0)
        prim_translate(
            camera_prim, (self.chase_camera_x_offset, 0.0, self.chase_camera_z_offset)
        )
        return camera_path

    @classmethod
    def build(cls, prim_path: str) -> "Robot":
        raise NotImplementedError

    def write_action(self, step_size: float):
        raise NotImplementedError

    def update_state(self):
        pos, ori = self.robot.get_local_pose()
        self.position.set_value(pos)
        self.orientation.set_value(ori)
        self.joint_positions.set_value(self.robot.get_joint_positions())
        self.joint_velocities.set_value(self.robot.get_joint_velocities())
        super().update_state()

    def write_replay_data(self):
        self.robot.set_local_pose(
            self.position.get_value(), self.orientation.get_value()
        )
        self.articulation_view.set_joint_positions(self.joint_positions.get_value())
        super().write_replay_data()

    def set_pose_2d(self, pose: Pose2d):
        self.articulation_view.initialize()
        self.robot.set_world_velocity(np.zeros(6))
        self.robot.post_reset()
        position, orientation = self.robot.get_local_pose()
        position[0], position[1], position[2] = pose.x, pose.y, self.z_offset
        orientation = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, pose.theta]))
        self.robot.set_local_pose(position, orientation)

    def get_pose_2d(self) -> Pose2d:
        position, orientation = self.robot.get_local_pose()
        theta = rot_utils.quats_to_euler_angles(orientation)[2]
        return Pose2d(x=position[0], y=position[1], theta=theta)


# =====================================================
#  DERIVED ROBOT CLASSES
# =====================================================


class WheeledRobot(Robot):
    wheel_dof_names: List[str]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float
    wheel_base: float

    def __init__(
        self,
        prim_path: str,
        robot: _WheeledRobot,
        articulation_view: _ArticulationView,
        controller: DifferentialController,
        main_camera: Sensor | None = None,
        fisheye_camera: Sensor | None = None,
        lidar_sensor: Sensor | None = None,
    ):
        super().__init__(
            prim_path,
            robot,
            articulation_view,
            main_camera,
            fisheye_camera,
            lidar_sensor,
        )
        self.controller = controller

    @classmethod
    def build(cls, prim_path: str) -> "WheeledRobot":
        stage = get_stage()
        world = get_world()
        # Explicitly add the USD reference first to ensure prims are created
        stage_add_usd_ref(stage=stage, path=prim_path, usd_path=cls.usd_url)
        robot = world.scene.add(
            _WheeledRobot(
                prim_path,
                wheel_dof_names=cls.wheel_dof_names,
            )
        )
        view = _ArticulationView(os.path.join(prim_path, cls.chassis_subpath))
        world.scene.add(view)
        controller = DifferentialController(
            name="controller", wheel_radius=cls.wheel_radius, wheel_base=cls.wheel_base
        )
        camera = cls.build_main_camera(prim_path)
        fisheye_camera = cls.build_fisheye_camera(prim_path) if hasattr(cls, 'fisheye_camera_base_path') else None
        lidar_sensor = cls.build_lidar_sensor(prim_path) if hasattr(cls, 'lidar_sensor_base_path') else None

        return cls(prim_path, robot, view, controller, camera, fisheye_camera, lidar_sensor)

    def write_action(self, step_size: float):
        self.robot.apply_wheel_actions(
            self.controller.forward(command=self.action.get_value())
        )


class FourWheeledRobot(Robot):
    wheel_dof_names: List[str]
    steering_dof_names: List[str]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float
    wheel_base: float
    track_width: float
    max_steer_angle: float = 0.6

    def __init__(
        self,
        prim_path: str,
        robot: _WheeledRobot,
        articulation_view: _ArticulationView,
        controller: AckermannController,
        front_camera: Optional[Any] = None,
    ):
        super().__init__(prim_path, robot, articulation_view, front_camera)
        self.controller = controller

    @classmethod
    def build(cls, prim_path: str) -> "FourWheeledRobot":
        stage = get_stage()
        world = get_world()
        # Explicitly add the USD reference first to ensure prims are created
        stage_add_usd_ref(stage=stage, path=prim_path, usd_path=cls.usd_url)
        try:
            robot = world.scene.add(
                _WheeledRobot(
                    prim_path,
                    wheel_dof_names=cls.wheel_dof_names,
                    steering_dof_names=cls.steering_dof_names,
                )
            )
        except TypeError:
            robot = world.scene.add(
                _WheeledRobot(
                    prim_path,
                    wheel_dof_names=cls.wheel_dof_names,
                )
            )
            if hasattr(robot, "set_steering_dof_names"):
                robot.set_steering_dof_names(cls.steering_dof_names)

        view = _ArticulationView(os.path.join(prim_path, cls.chassis_subpath))
        world.scene.add(view)
        controller = AckermannController(
            name="ackermann_controller",
            wheel_radius=cls.wheel_radius,
            wheel_base=cls.wheel_base,
            track_width=cls.track_width,
            max_steer_angle=cls.max_steer_angle,
        )
        camera = cls.build_main_camera(prim_path)
        return cls(prim_path, robot, view, controller, camera)

    def _parse_command(self, raw: Any) -> Tuple[float, float]:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        if isinstance(raw, dict):
            if "speed" in raw and "steer" in raw:
                return float(raw["speed"]), float(raw["steer"])
            if "v" in raw and "delta" in raw:
                return float(raw["v"]), float(raw["delta"])
        return 0.0, 0.0

    def write_action(self, step_size: float):
        v, delta = self._parse_command(self.action.get_value())
        if abs(delta) > self.max_steer_angle:
            delta = math.copysign(self.max_steer_angle, delta)
        controls = self.controller.forward(
            command={"speed": v, "steer": delta}, dt=step_size
        )
        self.robot.apply_wheel_actions(controls)


class IsaacLabRobot(Robot):
    usd_url: str
    articulation_path: str

    def __init__(
        self,
        prim_path: str,
        robot: _Robot,
        articulation_view: _ArticulationView,
        controller: Union[H1FlatTerrainPolicy, SpotFlatTerrainPolicy],
        front_camera: Sensor | None = None,
    ):
        super().__init__(prim_path, robot, articulation_view, front_camera)
        self.controller = controller

    @classmethod
    def build_policy(cls, prim_path: str):
        raise NotImplementedError

    @classmethod
    def build(cls, prim_path: str):
        stage = get_stage()
        world = get_world()
        stage_add_usd_ref(stage=stage, path=prim_path, usd_path=cls.usd_url)
        robot = _Robot(prim_path=prim_path)
        world.scene.add(robot)
        view = _ArticulationView(os.path.join(prim_path, cls.articulation_path))
        world.scene.add(view)
        controller = cls.build_policy(prim_path)
        prim = stage_get_prim(stage, prim_path)
        prim_translate(prim, (0, 0, cls.z_offset))
        camera = cls.build_main_camera(prim_path)
        return cls(prim_path, robot, view, controller, camera)

    def write_action(self, step_size):
        action = self.action.get_value()
        command = np.array([action[0], 0.0, action[1]])
        self.controller.forward(step_size, command)

    def set_pose_2d(self, pose):
        super().set_pose_2d(pose)
        self.controller.initialize()


# =====================================================
#  REGISTRY
# =====================================================

ROBOTS: Registry[Robot] = Registry()

# =====================================================
#  FINAL CLASSES (JetbotRobot, CarterRobot, H1Robot, SpotRobot, ForkliftCRobot, JetbotRobot_test…)
# =====================================================
# ... (your full set of robot classes here, unchanged except they all
#      benefit from build_front_camera now enabling RGB)

ROBOTS = Registry[Robot]()


@ROBOTS.register()
class JetbotRobot(WheeledRobot):
    physics_dt: float = 0.005

    z_offset: float = 0.1

    chase_camera_base_path = "chassis"
    chase_camera_x_offset: float = -0.5
    chase_camera_z_offset: float = 0.5
    chase_camera_tilt_angle: float = 60.0

    occupancy_map_radius: float = 0.25
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 0.5
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.25

    front_camera_base_path = "chassis/rgb_camera/front_hawk"
    front_camera_rotation = (0.0, 0.0, 0.0)
    front_camera_translation = (0.0, 0.0, 0.0)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 0.25
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 0.25
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 0.25)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 1.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 0.25
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    wheel_dof_names: List[str] = ["left_wheel_joint", "right_wheel_joint"]
    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Jetbot/jetbot.usd"
    chassis_subpath: str = "chassis"
    wheel_base: float = 0.1125
    wheel_radius: float = 0.03


@ROBOTS.register()
class CarterRobot(WheeledRobot):
    physics_dt: float = 0.005

    z_offset: float = 0.25

    chase_camera_base_path = "chassis_link"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.0

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "chassis_link/front_hawk/front_hawk"
    front_camera_rotation = (0.0, 0.0, 0.0)
    front_camera_translation = (0.0, 0.0, 0.0)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    wheel_dof_names: List[str] = ["joint_wheel_left", "joint_wheel_right"]
    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Carter/nova_carter_sensors.usd"
    chassis_subpath: str = "chassis_link"
    wheel_base = 0.413
    wheel_radius = 0.14


@ROBOTS.register()
class H1Robot(IsaacLabRobot):
    physics_dt: float = 0.005

    z_offset: float = 1.05

    chase_camera_base_path = "pelvis"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.0

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 2.0
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "d435_left_imager_link/front_camera/front"
    front_camera_rotation = (0.0, 250.0, 90.0)
    front_camera_translation = (-0.06, 0.0, 0.0)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/H1/h1.usd"
    articulation_path = "pelvis"
    controller_z_offset: float = 1.05

    @classmethod
    def build_policy(cls, prim_path: str):
        return H1FlatTerrainPolicy(
            prim_path=prim_path, position=np.array([0.0, 0.0, cls.controller_z_offset])
        )


@ROBOTS.register()
class SpotRobot(IsaacLabRobot):
    physics_dt: float = 0.005
    z_offset: float = 0.7

    chase_camera_base_path = "body"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.0

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    # front_camera_base_path = "body/front_camera"
    # front_camera_rotation = (180, 180, 180)
    # front_camera_translation = (0.44, 0.075, 0.01)
    # front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/BostonDynamics/spot/spot.usd"
    articulation_path = "/"
    controller_z_offset: float = 0.7

    @classmethod
    def build_policy(cls, prim_path: str):
        return SpotFlatTerrainPolicy(
            prim_path=prim_path, position=np.array([0.0, 0.0, cls.controller_z_offset])
        )


@ROBOTS.register()
class Jetbot_SCamera(WheeledRobot):
    physics_dt: float = 0.005
    z_offset: float = 0.1

    occupancy_map_radius: float = 0.25
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 0.5
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.25

# ===================== Sensores ================================
    chase_camera_base_path = "chassis/Sensors/"
    chase_camera_x_offset: float = -0.5
    chase_camera_z_offset: float = 0.5
    chase_camera_tilt_angle: float = 60.0

    main_camera_base_path = "chassis/Sensors/Camera"
    main_camera_rotation = (0.0, 0.0, 0.0)
    main_camera_translation = (0, 0, 1)
    main_camera_type = HawkCamera

    fisheye_camera_base_path = "chassis/Sensors/FisheyeCamera"
    fisheye_camera_rotation = (0.0, 0.0, 0.0)
    fisheye_camera_translation = (1, 0, 1)
    fisheye_camera_type = FisheyeCamera

    lidar_sensor_base_path = "chassis/Sensors/LidarSensor"
    lidar_sensor_rotation = (0.0, 0.0, 0.0)
    lidar_sensor_translation = (0, 0, 1)
    lidar_sensor_type = LidarSensor
    lidar_file_name: str = "HESAI_XT32_SD10"
    lidar_sensor_attributes: dict = {
        "horizontal_fov": 360.0,
        "vertical_fov": 30.0,
        "points_per_second": 100000,
        'omni:sensor:Core:scanRateBaseHz': 20,
        "range": 10.0,
        "channels": 32,
    }
   
# ===================== Ações ================================
    # ===== Teleop =====
    keyboard_linear_velocity_gain: float = 10.25
    keyboard_angular_velocity_gain: float = 30.0
    gamepad_linear_velocity_gain: float = 0.25
    gamepad_angular_velocity_gain: float = 1.0

    # ===== Random Action =====
    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 0.25)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 1.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    # ===== Path Following =====
    path_following_speed: float = 0.25
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    # ===== USD / rodas =====
    wheel_dof_names: List[str] = ["left_wheel_joint", "right_wheel_joint"]
    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Jetbot/jetbot.usd"
    chassis_subpath: str = "chassis"
    wheel_base: float = 0.1125
    wheel_radius: float = 0.03

    @classmethod
    def build(cls, prim_path: str) -> "Jetbot_SCamera":
        stage = get_stage()
        world = get_world()
        # Explicitly add the USD reference first to ensure prims are created
        stage_add_usd_ref(stage=stage, path=prim_path, usd_path=cls.usd_url)
        robot = world.scene.add(
            _WheeledRobot(
                prim_path,
                wheel_dof_names=cls.wheel_dof_names,
            )
        )
        view = _ArticulationView(os.path.join(prim_path, cls.chassis_subpath))
        world.scene.add(view)
        controller = DifferentialController(
            name="controller", wheel_radius=cls.wheel_radius, wheel_base=cls.wheel_base
        )
# ===================== buildando sensores ================================
        camera = cls.build_main_camera(prim_path)
        fisheye_camera = cls.build_fisheye_camera(prim_path)
        lidar_sensor = cls.build_lidar_sensor(prim_path)

        return cls(
            prim_path,
            robot,
            view,
            controller,
            camera,
            fisheye_camera,
            lidar_sensor,
        )
