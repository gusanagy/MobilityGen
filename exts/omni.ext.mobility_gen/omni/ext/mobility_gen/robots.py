# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations  # <-- DEIXE AQUI (linha 3), ou REMOVA em Python 3.10+
# Standard imports
import numpy as np
import os
import math
#from typing import List, Type, Tuple, Union
from typing import Optional, Sequence, Union, Dict, List, Tuple, Type, List



# Isaac Sim Imports
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.api.robots.robot import Robot as _Robot
from isaacsim.core.prims import Articulation as _ArticulationView
from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.stage import add_reference_to_stage
#from omni.isaac.core.utils.prims import set_world_pose
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# Extension imports
from omni.ext.mobility_gen.common import Buffer, Module
from omni.ext.mobility_gen.sensors import Sensor, HawkCamera
from omni.ext.mobility_gen.utils.global_utils import get_stage, get_world
from omni.ext.mobility_gen.utils.stage_utils import stage_get_prim, stage_add_camera, stage_add_usd_ref
from omni.ext.mobility_gen.utils.prim_utils import prim_rotate_x, prim_rotate_y, prim_rotate_z, prim_translate
from omni.ext.mobility_gen.types import Pose2d
from omni.ext.mobility_gen.utils.registry import Registry

# sensores novos (do seu sensors.py)
from omni.ext.mobility_gen.sensors import (
    HawkCamera,
    BevTopDownCamera,
    BevFrontDownCamera,
    RealSenseRGBDCamera,
    ZedStereoCamera,
)

# RELATIVOS (recomendado)
# from .registry import ROBOTS
# from .robot import Robot


# dentro da classe ForkliftC
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
import omni.usd
from omni.isaac.core.utils.rotations import euler_angles_to_quat



#=========================================================
#  BASE CLASSES
#=========================================================

#posso modificar a classe base para criar mais cameras e masi vistas das cameras 
# no momento nao vou me preocupar com os modelos das cameras apenas com conteudo depois colo os usds no local correto
class Robot(Module):
    """Abstract base class for robots

    This class defines an abstract base class for robots.

    Robot implementations must subclass this class, define the 
    required class parameters and abstract methods.

    The two main abstract methods subclasses must define are the build() and write_action()
    methods.
    
    Parameters:
        physics_dt (float): The physics time step the robot requires (in seconds).
            This may need to be modified depending on the underlying controller's
            
        z_offset (float): A z offset to use when spawning the robot to ensure it
            drops at an appropriate height when initializing.
        chase_camera_base_path (str):  A path (in the USD stage) relative to the 
            base path to use as a parent when defining the "chase" camera.  Typically,
            this is the same as the path to the transform used for determining
            the 2D pose of the robot.
        chase_camera_x_offset (str):  The x offset of the chase camera.  Typically this is
            negative, to locate the camera behind the robot.
        chase_camera_z_offset (str): The z offset of the chase camera.  Typically this is
            positive, to locate the camera above the robot.
        chase_camera_tile_angle (float):  The tilt angle of the chase camera.  Typically
            this does not need to be modified.
        front_camera_type (Type[Sensor]):  The configurable sensor class to attach
            at the front camera for the robot.  This should be a final sensor class (like HawkCamera)
            that can be built using the class method HawkCamera.build(prim_path).
        front_camera_base_path (str):  The relative path (in the USD stage) relative to the 
            robot prim path to use as the basis for creating the front camera XForm.
        front_camera_rotation (Tuple[float, float, float]):  The (x, y, z) rotation to apply
            when building the XForm for the front camera.
        front_camera_translation (Tuple[float, float, float]):  The (x, y, z) rotation to apply
            when building the XForm for the front camera.

    """

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

    front_camera_type: Type[Sensor]
    front_camera_base_path: str
    front_camera_rotation: Tuple[float, float, float]
    front_camera_translation: Tuple[float, float, float]

    keyboard_linear_velocity_gain: float
    keyboard_angular_velocity_gain: float

    gamepad_linear_velocity_gain: float
    gamepad_angular_velocity_gain: float

    random_action_linear_velocity_range: Tuple[float, float]
    random_action_angular_velocity_range: Tuple[float, float]
    random_action_linear_acceleration_std: float
    random_action_angular_acceleration_std: float
    random_action_grid_pose_sampler_grid_size: float


    path_following_speed: float
    path_following_angular_gain: float
    path_following_stop_distance_threshold: float
    path_following_forward_angle_threshold = math.pi
    path_following_target_point_offset_meters: float


    def __init__(self,
            prim_path: str,
            robot: _Robot,
            articulation_view: _ArticulationView,
            front_camera: Sensor
        ):
        self.prim_path = prim_path
        self.robot = robot
        self.articulation_view = articulation_view

        self.action = Buffer(np.zeros(2))
        self.position = Buffer()
        self.orientation = Buffer()
        self.joint_positions = Buffer()
        self.joint_velocities = Buffer()
        self.front_camera = front_camera

    @classmethod
    def build_front_camera(cls, prim_path):
        
        # Add camera
        camera_path = os.path.join(prim_path, cls.front_camera_base_path)
        front_camera_xform = XFormPrim(camera_path)

        stage = get_stage()
        front_camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(front_camera_prim, cls.front_camera_rotation[0])
        prim_rotate_y(front_camera_prim, cls.front_camera_rotation[1])
        prim_rotate_z(front_camera_prim, cls.front_camera_rotation[2])
        prim_translate(front_camera_prim, cls.front_camera_translation)

        return cls.front_camera_type.build(prim_path=camera_path)

    def build_chase_camera(self) -> str:

        stage = get_stage()

        camera_path = os.path.join(self.prim_path, self.chase_camera_base_path, "chase_camera")
        stage_add_camera(stage, 
            camera_path, 
            focal_length=10, horizontal_aperature=30, vertical_aperature=30
        )
        camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(camera_prim, self.chase_camera_tilt_angle)
        prim_rotate_y(camera_prim, 0)
        prim_rotate_z(camera_prim, -90)
        prim_translate(camera_prim, (self.chase_camera_x_offset, 0., self.chase_camera_z_offset))

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
            self.position.get_value(),
            self.orientation.get_value()
        )
        self.articulation_view.set_joint_positions(
            self.joint_positions.get_value()
        )
        super().write_replay_data()

    def set_pose_2d(self, pose: Pose2d):
        self.articulation_view.initialize()
        self.robot.set_world_velocity(np.array([0., 0., 0., 0., 0., 0.]))
        self.robot.post_reset()
        position, orientation = self.robot.get_local_pose()
        position[0] = pose.x
        position[1] = pose.y
        position[2] = self.z_offset
        orientation = rot_utils.euler_angles_to_quats(np.array([0., 0., pose.theta]))
        self.robot.set_local_pose(
            position, orientation
        )
    
    def get_pose_2d(self) -> Pose2d:
        position, orientation = self.robot.get_local_pose()
        theta = rot_utils.quats_to_euler_angles(orientation)[2]
        return Pose2d(
            x=position[0],
            y=position[1],
            theta=theta
        )
    

class WheeledRobot(Robot):

    # Wheeled robot parameters
    wheel_dof_names: List[str]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float
    wheel_base: float

    def __init__(self,
            prim_path: str,
            robot: _WheeledRobot,
            articulation_view: _ArticulationView,
            controller: DifferentialController,
            front_camera: Sensor | None = None
        ):
        super().__init__(
            prim_path=prim_path,
            robot=robot,
            articulation_view=articulation_view,
            front_camera=front_camera
        )
        self.controller = controller
        self.robot = robot
        
    @classmethod
    def build(cls, prim_path: str) -> "WheeledRobot":

        world = get_world()

        robot = world.scene.add(_WheeledRobot(
            prim_path,
            wheel_dof_names=cls.wheel_dof_names,
            create_robot=True,
            usd_path=cls.usd_url
        ))

        view = _ArticulationView(
            os.path.join(prim_path, cls.chassis_subpath)
        )

        world.scene.add(view)

        controller = DifferentialController(
            name="controller",
            wheel_radius=cls.wheel_radius,
            wheel_base=cls.wheel_base
        )
        
        camera = cls.build_front_camera(prim_path)

        return cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=view,
            controller=controller,
            front_camera=camera
        )
    
    def write_action(self, step_size: float):
        self.robot.apply_wheel_actions(
            self.controller.forward(
                command=self.action.get_value()
            )
        )
from typing import List, Tuple, Dict, Any, Optional
import os

from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot as _WheeledRobot
from isaacsim.core.prims import ArticulationView as _ArticulationView
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

from omni.isaac.core.world import get_world
from .base_robot import Robot  # ajuste o import conforme seu projeto


class FourWheeledRobot(Robot):

    # Wheeled robot parameters (defina nas subclasses concretas)
    wheel_dof_names: List[str]        # ex: ["fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"]
    steering_dof_names: List[str]     # ex: ["fl_steer_joint", "fr_steer_joint"]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float               # raio da roda (m)
    wheel_base: float                 # distância entre eixos (m)
    track_width: float                # bitola (distância entre rodas esquerda-direita do mesmo eixo) (m)
    max_steer_angle: float = 0.6      # limite de esterçamento (rad), ajuste conforme seu veículo

    def __init__(
        self,
        prim_path: str,
        robot: _WheeledRobot,
        articulation_view: _ArticulationView,
        controller: AckermannController,
        front_camera: Optional[Any] = None
    ):
        super().__init__(
            prim_path=prim_path,
            robot=robot,
            articulation_view=articulation_view,
            front_camera=front_camera
        )
        self.controller = controller
        self.robot = robot

    @classmethod
    def build(cls, prim_path: str) -> "FourWheeledRobot":
        world = get_world()

        # Cria o robô com juntas de rodas e (se suportado) de direção
        # Algumas versões aceitam steering_dof_names no ctor; deixamos fallback para compatibilidade
        try:
            robot = world.scene.add(_WheeledRobot(
                prim_path,
                wheel_dof_names=cls.wheel_dof_names,
                steering_dof_names=cls.steering_dof_names,
                create_robot=True,
                usd_path=cls.usd_url
            ))
        except TypeError:
            robot = world.scene.add(_WheeledRobot(
                prim_path,
                wheel_dof_names=cls.wheel_dof_names,
                create_robot=True,
                usd_path=cls.usd_url
            ))
            # fallback se a API não aceitar no construtor
            if hasattr(robot, "set_steering_dof_names"):
                robot.set_steering_dof_names(cls.steering_dof_names)

        # ArticulationView do chassi (mantém estrutura original)
        view = _ArticulationView(os.path.join(prim_path, cls.chassis_subpath))
        world.scene.add(view)

        # Controlador Ackermann (mantém “controller” na estrutura)
        controller = AckermannController(
            name="ackermann_controller",
            wheel_radius=cls.wheel_radius,
            wheel_base=cls.wheel_base,
            track_width=cls.track_width,
            max_steer_angle=cls.max_steer_angle
        )

        camera = cls.build_front_camera(prim_path)

        return cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=view,
            controller=controller,
            front_camera=camera
        )

    def _parse_command(self, raw_cmd: Any) -> Tuple[float, float]:
        """
        Converte o comando vindo de self.action.get_value() para (speed, steer).
        Aceita:
          - tupla/lista: (speed, steer)
          - dict com chaves 'speed'/'steer' ou 'v'/'delta'
        speed em m/s; steer em rad (ângulo da roda dianteira).
        """
        if isinstance(raw_cmd, (list, tuple)) and len(raw_cmd) >= 2:
            return float(raw_cmd[0]), float(raw_cmd[1])
        if isinstance(raw_cmd, dict):
            if "speed" in raw_cmd and "steer" in raw_cmd:
                return float(raw_cmd["speed"]), float(raw_cmd["steer"])
            if "v" in raw_cmd and "delta" in raw_cmd:
                return float(raw_cmd["v"]), float(raw_cmd["delta"])
        # default seguro
        return 0.0, 0.0

    def write_action(self, step_size: float):
        """
        Usa AckermannController para gerar:
          - velocidades das quatro rodas [FL, FR, RL, RR]
          - ângulos de direção para as juntas dianteiras [FL_steer, FR_steer]
        e aplica via WheeledRobot.apply_wheel_actions(...).
        """
        raw = self.action.get_value()
        speed, steer = self._parse_command(raw)

        # Clampeia ângulo ao limite do veículo
        if abs(steer) > self.max_steer_angle:
            steer = self.max_steer_angle if steer > 0 else -self.max_steer_angle

        # O controller.forward retorna um payload compatível com apply_wheel_actions
        # (normalmente contendo wheel_velocity_commands e steering_commands)
        controls = self.controller.forward(
            command={"speed": speed, "steer": steer},
            dt=step_size
        )

        # Mantém o padrão original: passa o pacote único para apply_wheel_actions
        self.robot.apply_wheel_actions(controls)



class IsaacLabRobot(Robot):

    usd_url: str
    articulation_path: str

    def __init__(self, 
            prim_path: str, 
            robot: _Robot,
            articulation_view: _ArticulationView,
            controller: Union[H1FlatTerrainPolicy, SpotFlatTerrainPolicy],
            front_camera: Sensor | None = None
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

        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url
        )
        
        robot = _Robot(prim_path=prim_path)

        world.scene.add(robot)

        # Articulation
        view = _ArticulationView(
            os.path.join(prim_path, cls.articulation_path)
        )

        world.scene.add(view)

        # Controller
        controller = cls.build_policy(prim_path)

        prim = stage_get_prim(stage, prim_path)        
        prim_translate(prim, (0, 0, cls.z_offset))


        camera = cls.build_front_camera(prim_path)

        return cls(
            prim_path=prim_path, 
            robot=robot, 
            articulation_view=view, 
            controller=controller,
            front_camera=camera
        )
    
    def write_action(self, step_size):
        action = self.action.get_value()
        command = np.array([action[0], 0., action[1]])
        self.controller.forward(step_size, command)

    def set_pose_2d(self, pose):
        super().set_pose_2d(pose)
        self.controller.initialize()


#=========================================================
#  FINAL CLASSES
#=========================================================

ROBOTS = Registry[Robot]()


@ROBOTS.register()
class JetbotRobot(WheeledRobot):

    physics_dt: float = 0.005

    z_offset: float = 0.1

    chase_camera_base_path = "chassis"
    chase_camera_x_offset: float = -0.5
    chase_camera_z_offset: float = 0.5
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 0.25
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 0.5
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.25

    front_camera_base_path = "chassis/rgb_camera/front_hawk"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (0., 0., 0.)
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
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "chassis_link/front_hawk/front_hawk"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (0., 0., 0.)
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
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 2.0
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "d435_left_imager_link/front_camera/front"
    front_camera_rotation = (0., 250., 90.)
    front_camera_translation = (-0.06, 0., 0.)
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
            prim_path=prim_path,
            position=np.array([0., 0., cls.controller_z_offset])
        )


@ROBOTS.register()
class SpotRobot(IsaacLabRobot):

    physics_dt: float = 0.005
    z_offset: float = 0.7

    chase_camera_base_path = "body"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "body/front_camera"
    front_camera_rotation = (180, 180, 180)
    front_camera_translation = (0.44, 0.075, 0.01)
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
    path_following_stop_distance_threshold:1.0

    @classmethod
    def build_policy(cls, prim_path: str):
        return SpotFlatTerrainPolicy(
            prim_path=prim_path,
            position=np.array([0., 0., cls.controller_z_offset])
        )
    
    ####################################################################################
    #Colocar aqui a nova definição do robo forklift# Preliminar faltam ajustes e testes#
    ####################################################################################

import math
from typing import List, Tuple
from omni.isaac.core.utils.viewports import set_camera_view

from .four_wheeled_robot import FourWheeledRobot  # <- importe a superclasse que criamos
from .sensors.hawk_camera import HawkCamera       # ajuste conforme seu projeto
from .registry import ROBOTS                      # mantém seu registry/decorator

@ROBOTS.register()
class ForkliftCRobot(FourWheeledRobot):
    # ====== Timing / Física ======
    physics_dt: float = 0.005

    # ====== Pose / Offsets ======
    z_offset: float = 0.5

    # ====== Câmera de perseguição (se você usa) ======
    chase_camera_base_path = "body"  # root do chassi
    chase_camera_x_offset: float = -1.6
    chase_camera_z_offset: float = 5.0
    chase_camera_tilt_angle: float = 55.0

    # ====== Mapa de ocupação (mantidos) ======
    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    # ====== Câmera frontal (mantidos) ======
    front_camera_base_path = "sensors/rgb_camera/front_hawk"
    front_camera_rotation = (0.0, 0.0, 0.0)
    front_camera_translation = (0.55, 0.0, 1.40)
    front_camera_type = HawkCamera

    # ====== Ganhos de teleop (mantidos) ======
    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0
    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    # ====== Ação aleatória (mantidos) ======
    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 2.0
    random_action_angular_acceleration_std: float = 2.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    # ====== Path following (mantidos) ======
    path_following_speed: float = 0.8
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.3
    path_following_forward_angle_threshold = math.pi
    path_following_target_point_offset_meters: float = 0.0

    # ====== Parâmetros do robô / USD ======
    # Quatro juntas de roda (ordem esperada pelo controlador: FL, FR, RL, RR)
    # Ajuste os nomes para bater com o seu forklift_c.usd.
    wheel_dof_names: List[str] = [
        "left_front_wheel_joint",
        "right_front_wheel_joint",
        "left_back_wheel_joint",
        "right_back_wheel_joint",
    ]

    # Duas juntas de direção (Ackermann) — aqui assumimos direção dianteira
    # Se o seu modelo tiver direção traseira, troque para, por ex.:
    # ["left_back_steer_joint", "right_back_steer_joint"]
    steering_dof_names: List[str] = [
        "left_front_steer_joint",
        "right_front_steer_joint",
    ]

    usd_url: str = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
    )
    chassis_subpath: str = "body"  # articulation no root

    # Geometria do veículo (use valores do seu modelo se souber)
    wheel_base: float = 1.10     # distância entre eixos (m)
    track_width: float = 0.95    # bitola (m) — ajuste conforme seu USD
    wheel_radius: float = 0.50   # raio (m)
    max_steer_angle: float = 0.60  # limite de esterçamento (rad)

    # Observação:
    # - A superclasse FourWheeledRobot provê .build(...) que cria o WheeledRobot,
    #   ArticulationView e o AckermannController usando os atributos acima.
    # - Se os nomes de DOF diferirem no seu USD, basta corrigi-los aqui.


# @ROBOTS.register()
# class ForkliftCRobot(WheeledRobot):

#     physics_dt: float = 0.005

#     z_offset: float = 0.5

#     chase_camera_base_path = "body"  # root
#     chase_camera_x_offset: float = -1.6
#     chase_camera_z_offset: float = 5.0
#     chase_camera_tilt_angle: float = 55.0

#     # occupancy_map_radius: float = 1.5
#     # occupancy_map_z_min: float = -0.2
#     # occupancy_map_z_max: float = 3.0
#     # occupancy_map_cell_size: float = 0.05
#     # occupancy_map_collision_radius: float = 0.6

#     occupancy_map_radius: float = 1.0
#     occupancy_map_z_min: float = 0.1
#     occupancy_map_z_max: float = 0.62
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.5

#     front_camera_base_path = "sensors/rgb_camera/front_hawk"
#     front_camera_rotation = (0.0, 0.0, 0.0)
#     front_camera_translation = (0.55, 0.0, 1.40)
#     front_camera_type = HawkCamera

#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0

#     gamepad_linear_velocity_gain: float = 1.0
#     gamepad_angular_velocity_gain: float = 1.0

#     random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
#     random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
#     random_action_linear_acceleration_std: float = 2.0
#     random_action_angular_acceleration_std: float = 2.0
#     random_action_grid_pose_sampler_grid_size: float = 5.0

#     path_following_speed: float = 0.8
#     path_following_angular_gain: float = 1.0
#     path_following_stop_distance_threshold: float = 0.3
#     path_following_forward_angle_threshold = math.pi
#     path_following_target_point_offset_meters: float = 0.0

#     #wheel_dof_names: List[str] = ["left_back_wheel_joint", "right_back_wheel_joint", "left_front_wheel_joint", "right_front_wheel_joint"]
#     wheel_dof_names: List[str] = ["left_back_wheel_joint", "right_back_wheel_joint"]
#     usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
#     chassis_subpath: str = "body"  # articulation no root
#     wheel_base: float = 1.10
#     wheel_radius: float = 0.5


# @ROBOTS.register()
# class ForkliftCRobot(WheeledRobot):
#     # Passo de física
#     physics_dt: float = 1.0 / 120.0

#     # Spawn
#     z_offset: float = 0.10

#     # Chase-cam (apenas visual)
#     chase_camera_base_path: str = ""   # no root
#     chase_camera_x_offset: float = -1.6
#     chase_camera_z_offset: float = 1.2
#     chase_camera_tilt_angle: float = 55.0

#     # Ocupância (típico de empilhadeira)
#     occupancy_map_radius: float = 1.5
#     occupancy_map_z_min: float = -0.2
#     occupancy_map_z_max: float = 3.0
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.6

#     # Câmera frontal oficial (MobilityGen)
#     front_camera_base_path: str = "sensors/front_hawk"
#     front_camera_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
#     front_camera_translation: Tuple[float, float, float] = (0.55, 0.0, 1.40)
#     front_camera_type = HawkCamera

#     # Ganhos teleop
#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0
#     gamepad_linear_velocity_gain: float = 1.0
#     gamepad_angular_velocity_gain: float = 1.0

#     # Ação aleatória / path following (compatibilidade)
#     random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
#     random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
#     random_action_linear_acceleration_std: float = 2.0
#     random_action_angular_acceleration_std: float = 2.0
#     random_action_grid_pose_sampler_grid_size: float = 5.0

#     path_following_speed: float = 0.8
#     path_following_angular_gain: float = 1.0
#     path_following_stop_distance_threshold: float = 0.3
#     path_following_forward_angle_threshold = math.pi
#     path_following_target_point_offset_meters: float = 0.0

#     # === Isaac assets ===
#     usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
#     # SE usar outro asset (S3 4.2) troque aqui para o mesmo que já carregava OK no seu setup.

#     # O root da articulação é o próprio /World/robot
#     chassis_subpath: str = ""   # <<<<<< chave para resolver seu erro

#     # Cinemática aproximada para controller diferencial (apenas traseiras motrizes)
#     wheel_base: float = 1.10    # distância entre eixos (m)
#     wheel_radius: float = 0.165 # raio da roda traseira (m)

#     # Nomes EXATOS das rodas traseiras no seu USD:
#     wheel_dof_names: List[str] = [
#         "rear_left_wheel_joint",   # TODO: troque pelo nome real
#         "rear_right_wheel_joint",  # TODO: troque pelo nome real
#     ]
#     # ====== build overriden p/ garantir IMU funcional ======
#     # @classmethod
#     # def build(cls, prim_path: str) -> "ForkliftCRobot":
#     #     # cria robô, articulation view, controller e câmera (lógica do WheeledRobot)
#     #     self = super().build(prim_path)

#     #     # >>> IMU FIX: garantir que o mount da câmera frontal tenha RigidBodyAPI <<<
#     #     # Muitos pipelines criam a IMU como filho do mount da câmera; o pai precisa ser rigid.
#     #     stage = get_stage()
#     #     cam_mount_path = os.path.join(prim_path, cls.front_camera_base_path)
#     #     cam_mount_prim = stage_get_prim(stage, cam_mount_path)
#     #     if cam_mount_prim and cam_mount_prim.IsValid():
#     #         # aplica PhysxRigidBodyAPI se ainda não tiver
#     #         if not PhysxSchema.PhysxRigidBodyAPI(cam_mount_prim):
#     #             PhysxSchema.PhysxRigidBodyAPI.Apply(cam_mount_prim)
#     #         # (opcional) manter como “leve”: sem colisor, só para fornecer velocidades/pose à IMU

#     #     # pronto — agora, se o seu script/extension cria “.../Imu_Sensor” sob esse mount,
#     #     # o parent já tem RigidBody e a IMU inicializa sem warnings.

#     #     return self

