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
import os
import math
from typing import Tuple, Optional

import numpy as np
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from pxr import UsdGeom, Gf, Sdf
from isaacsim.core.prims import SingleXFormPrim as XFormPrim

from omni.ext.mobility_gen.utils.global_utils import get_stage
from omni.ext.mobility_gen.utils.stage_utils import stage_add_usd_ref
from omni.ext.mobility_gen.common import Module, Buffer

class Sensor(Module):

    def build(self, prim_path: str):
        raise NotImplementedError
    
    def attach(self, prim_path: str):
        raise NotImplementedError

class Camera(Sensor):

    def __init__(self,
            prim_path: str,
            resolution: Tuple[int, int]
        ):

        self._prim_path = prim_path
        self._resolution = resolution
        self._render_product = None
        self._rgb_annotator = None
        self._segmentation_annotator = None
        self._instance_id_segmentation_annotator = None
        self._normals_annotator = None
        self._depth_annotator = None
        self._xform_prim = XFormPrim(self._prim_path)

        self.rgb_image = Buffer(tags=["rgb"])
        self.segmentation_image = Buffer(tags=["segmentation"])
        self.segmentation_info = Buffer()
        self.depth_image = Buffer(tags=["depth"])
        self.instance_id_segmentation_image = Buffer(tags=["segmentation"])
        self.instance_id_segmentation_info = Buffer()
        self.normals_image = Buffer(tags=['normals'])
        self.position = Buffer()
        self.orientation = Buffer()

    def enable_rendering(self):
        
        self._render_product = rep.create.render_product(
            self._prim_path,
            self._resolution,
            force_new=False
        )

    def disable_rendering(self):
        if self._render_product is None:
            return
        
        if self._rgb_annotator is not None:
            self._rgb_annotator.detach()
            self._rgb_annotator = None
        
        if self._segmentation_annotator is not None:
            self._segmentation_annotator.detach()
            self._segmentation_annotator = None

        if self._depth_annotator is not None:
            self._depth_annotator.detach()
            self._depth_annotator = None

        self._render_product.destroy()
        self._render_product = None
    
    def enable_rgb_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._rgb_annotator is not None:
            return
        try:
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
            self._rgb_annotator.attach(self._render_product)
        except Exception as e:
            print(f"[Camera] enable_rgb_rendering failed: {e}")

    def enable_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._segmentation_annotator is not None:
            return
        try:
            self._segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params=dict(colorize=False)
            )
            self._segmentation_annotator.attach(self._render_product)
        except Exception as e:
            print(f"[Camera] enable_segmentation_rendering failed: {e}")

    def enable_instance_id_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._instance_id_segmentation_annotator is not None:
            return
        try:
            self._instance_id_segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation", init_params=dict(colorize=False)
            )
            self._instance_id_segmentation_annotator.attach(self._render_product)
        except Exception as e:
            print(f"[Camera] enable_instance_id_segmentation_rendering failed: {e}")

    def enable_depth_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._depth_annotator is not None:
            return
        try:
            self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
                "distance_to_camera"
            )
            self._depth_annotator.attach(self._render_product)
        except Exception as e:
            print(f"[Camera] enable_depth_rendering failed: {e}")

    def enable_normals_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._normals_annotator is not None:
            return
        try:
            self._normals_annotator = rep.AnnotatorRegistry.get_annotator(
                "normals"
            )
            self._normals_annotator.attach(self._render_product)
        except Exception as e:
            print(f"[Camera] enable_normals_rendering failed: {e}")

    def update_state(self):
        if self._rgb_annotator is not None:
            rgb_data = self._rgb_annotator.get_data()
            # Verificar se os dados estão prontos (array 3D com shape HxWxC)
            if rgb_data is not None and rgb_data.ndim >= 3:
                self.rgb_image.set_value(rgb_data[:, :, :3])
        if self._segmentation_annotator is not None:
            data = self._segmentation_annotator.get_data()
            seg_image = data['data']
            seg_info = data['info']
            self.segmentation_image.set_value(seg_image)
            self.segmentation_info.set_value(seg_info)

        if self._depth_annotator is not None:
            self.depth_image.set_value(
                self._depth_annotator.get_data()
            )

        if self._instance_id_segmentation_annotator is not None:
            data = self._instance_id_segmentation_annotator.get_data()
            id_seg_image = data['data']
            id_seg_info = data['info']
            self.instance_id_segmentation_image.set_value(id_seg_image)
            self.instance_id_segmentation_info.set_value(id_seg_info)

        if self._normals_annotator is not None:
            data = self._normals_annotator.get_data()
            self.normals_image.set_value(data)
            
        position, orientation = self._xform_prim.get_world_pose()
        self.position.set_value(position)
        self.orientation.set_value(orientation)
        
        super().update_state()


#=========================================================
#  FINAL CLASSES
#=========================================================
class HawkCamera(Sensor):

    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Sensors/LeopardImaging/Hawk/hawk_v1.1_nominal.usd"
    resolution: Tuple[int, int] = (960, 600)
    left_camera_path: str = "left/camera_left"
    right_camera_path: str = "right/camera_right"

    def __init__(self, 
            left: Camera, 
            right: Camera
        ):
        self.left = left
        self.right = right
    
    @classmethod
    def build(cls, prim_path: str) -> "HawkCamera":
        
        stage = get_stage()

        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url
        )

        return cls.attach(prim_path)
    
    @classmethod
    def attach(cls, prim_path: str) -> "HawkCamera":
        
        left_camera = Camera(os.path.join(prim_path, cls.left_camera_path), cls.resolution)
        right_camera = Camera(os.path.join(prim_path, cls.right_camera_path), cls.resolution)

        return HawkCamera(left_camera, right_camera)
    


# ================================================
# ZED Stereo em formato HawkCamera
# ================================================
class ZedStereoCamera(Sensor):
    """
    Par estéreo estilo ZED com o mesmo padrão estrutural de HawkCamera:
    - Atributos de classe: usd_url, resolution, left/right paths
    - build(): injeta a referência USD e delega para attach()
    - attach(): empacota os dois prims internos em duas Cameras (left/right)
    """
    # Ajuste para o caminho real do asset no seu servidor/Omniverse
    usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc"
    resolution: Tuple[int, int] = (1280, 720)
    left_camera_path: str = "left/camera_left"     # confirme no USD
    right_camera_path: str = "right/camera_right"  # confirme no USD

    def __init__(self, left: Camera, right: Camera):
        self.left = left
        self.right = right

    @classmethod
    def build(cls, prim_path: str) -> "ZedStereoCamera":
        """
        Cria um Xform em `prim_path`, adiciona a referência USD da ZED
        e retorna o wrapper estéreo com left/right.
        """
        stage = get_stage()

        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url,
        )

        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "ZedStereoCamera":
        """
        Resolve os caminhos absolutos dos prims internos de cada câmera
        (ex.: "<prim_path>/left/camera_left" e "<prim_path>/right/camera_right"),
        empacota em duas instâncias Camera e retorna o par estéreo.
        """
        left_camera = Camera(os.path.join(prim_path, cls.left_camera_path), cls.resolution)
        right_camera = Camera(os.path.join(prim_path, cls.right_camera_path), cls.resolution)
        return ZedStereoCamera(left_camera, right_camera)


# =========================================================
# Helpers de câmera (como no seu código original)
# =========================================================

def _define_camera_prim(path: str) -> UsdGeom.Camera:
    """Garante/define um prim de câmera no path e o retorna."""
    stage = omni.usd.get_context().get_stage()
    return UsdGeom.Camera.Define(stage, path)

def _xform_translate(path: str, xyz: Tuple[float, float, float]):
    """Aplica (ou cria) um XformOp de Translate no prim path."""
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = xf.GetOrderedXformOps()
    t_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
            break
    if t_op is None:
        t_op = xf.AddTranslateOp()
    attr = t_op.GetAttr() if hasattr(t_op, "GetAttr") else t_op.GetOpAttr()
    typ = str(attr.GetTypeName())
    if "float3" in typ or "GfVec3f" in typ:
        attr.Set(Gf.Vec3f(*[float(v) for v in xyz]))
    else:
        attr.Set(Gf.Vec3d(*[float(v) for v in xyz]))

def _xform_orient_quat(path: str, quat_wxyz: Tuple[float, float, float, float]):
    """Aplica (ou cria) um XformOp de Orient (quaternion w,x,y,z) no prim path."""
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = xf.GetOrderedXformOps()
    o_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            o_op = op
            break
    if o_op is None:
        o_op = xf.AddOrientOp()
    qw, qx, qy, qz = [float(q) for q in quat_wxyz]
    attr = o_op.GetAttr() if hasattr(o_op, "GetAttr") else o_op.GetOpAttr()
    typ = str(attr.GetTypeName())
    if "quatf" in typ or "GfQuatf" in typ:
        attr.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    else:
        attr.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))

def _quat_from_euler_xyz(rx_deg: float, ry_deg: float, rz_deg: float):
    """Converte Euler XYZ (graus) para quaternion (w, x, y, z) com composição intrínseca."""
    rx = math.radians(rx_deg); ry = math.radians(ry_deg); rz = math.radians(rz_deg)
    cx, cy, cz = math.cos(rx/2), math.cos(ry/2), math.cos(rz/2)
    sx, sy, sz = math.sin(rx/2), math.sin(ry/2), math.sin(rz/2)
    qw = cx*cy*cz - sx*sy*sz
    qx = sx*cy*cz + cx*sy*sz
    qy = cx*sy*cz - sx*cy*sz
    qz = cx*cy*sz + sx*sy*cz
    return (qw, qx, qy, qz)


# =========================================================
# Câmera Fisheye com distorção OpenCV
# =========================================================
class FisheyeCamera(Sensor):
    """
    Câmera fisheye usando modelo de distorção OpenCV fisheye.
    Baseado na documentação do Isaac Sim 5.1:
    https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html
    """
    resolution: Tuple[int, int] = (1920, 1200)
    camera_matrix = [[455.8, 0.0, 943.8], [0.0, 454.7, 602.3], [0.0, 0.0, 1.0]]
    distortion_coefficients = [0.05, 0.01, -0.003, -0.0005]
    pixel_size = 3
    f_stop = 1.8
    focus_distance = 1.5

    right_camera_path: str = "right/camera_right"  # para compatibilidade com HawkCamera
    left_camera_path: str = "left/camera_left"    # para compatibilidade com HawkCamera

    def __init__(self, left: Camera, right: Camera):
        self.left = left
        self.right = right
        
    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        resolution: Tuple[int, int] = None,
        camera_matrix: list = None,
        distortion_coefficients: list = None,
        pixel_size: float = None,
        f_stop: float = None,
        focus_distance: float = None,
        near: float = 0.05,
        far: float = 100000.0,
    ) -> "FisheyeCamera":
        """
        Cria uma câmera fisheye com parâmetros de calibração OpenCV.
        
        Args:
            prim_path: Caminho USD onde a câmera será criada
            resolution: Tupla (largura, altura) da imagem
            camera_matrix: Matriz intrínseca 3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            distortion_coefficients: Lista [k1, k2, k3, k4] de distorção fisheye
            pixel_size: Tamanho do pixel em microns
            f_stop: f-number para depth of field (0.0 desabilita DoF)
            focus_distance: Distância de foco em metros
            near: Plano de clipping próximo
            far: Plano de clipping distante
        """
        res = resolution or cls.resolution
        width, height = res
        cam_matrix = camera_matrix or cls.camera_matrix
        dist_coeffs = distortion_coefficients or cls.distortion_coefficients
        px_size = pixel_size or cls.pixel_size
        fstop = f_stop or cls.f_stop
        focus_dist = focus_distance or cls.focus_distance

        ((fx, _, cx), (_, fy, cy), (_, _, _)) = cam_matrix

        horizontal_aperture = px_size * width * 1e-6  
        vertical_aperture = px_size * height * 1e-6   
        focal_length_x = px_size * fx * 1e-6          
        focal_length_y = px_size * fy * 1e-6          
        focal_length = (focal_length_x + focal_length_y) / 2 

        cam_prim = _define_camera_prim(prim_path)
        cam_prim.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
        cam_prim.CreateFocalLengthAttr(focal_length * 1000.0)  
        cam_prim.CreateHorizontalApertureAttr(horizontal_aperture * 1000.0)
        cam_prim.CreateVerticalApertureAttr(vertical_aperture * 1000.0)  
        cam_prim.CreateFStopAttr(fstop)
        cam_prim.CreateFocusDistanceAttr(focus_dist)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        # Aplicar atributos de distorção fisheye OpenCV
        # Usando os atributos do schema OmniLensDistortion
        prim.CreateAttribute("omni:lens:distortion:model", Sdf.ValueTypeNames.Token).Set("OpenCVFisheye")
        prim.CreateAttribute("omni:lens:distortion:opencv:fisheye:k1", Sdf.ValueTypeNames.Float).Set(float(dist_coeffs[0]))
        prim.CreateAttribute("omni:lens:distortion:opencv:fisheye:k2", Sdf.ValueTypeNames.Float).Set(float(dist_coeffs[1]))
        prim.CreateAttribute("omni:lens:distortion:opencv:fisheye:k3", Sdf.ValueTypeNames.Float).Set(float(dist_coeffs[2]))
        prim.CreateAttribute("omni:lens:distortion:opencv:fisheye:k4", Sdf.ValueTypeNames.Float).Set(float(dist_coeffs[3]))
        prim.CreateAttribute("omni:lens:distortion:opencv:cx", Sdf.ValueTypeNames.Float).Set(float(cx))
        prim.CreateAttribute("omni:lens:distortion:opencv:cy", Sdf.ValueTypeNames.Float).Set(float(cy))
        prim.CreateAttribute("omni:lens:distortion:opencv:fx", Sdf.ValueTypeNames.Float).Set(float(fx))
        prim.CreateAttribute("omni:lens:distortion:opencv:fy", Sdf.ValueTypeNames.Float).Set(float(fy))

        return cls.attach(prim_path, res)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "FisheyeCamera":
        """
        Anexa a uma câmera fisheye existente.
        """
        left_camera = Camera(prim_path, resolution or cls.resolution)
        right_camera = Camera(prim_path, resolution or cls.resolution)

        return FisheyeCamera(left_camera, right_camera)  # Usamos ambos os cameras para compatibilidade com HawkCamera
# =========================================================

#=========================================================
# Lidar Sensor
#=========================================================
class LidarSensor(Sensor):
    """
    RTX Lidar sensor using the LidarRtx class from Isaac Sim 5.1.0.
    Uses OmniLidar prims with OmniSensorGenericLidarCoreAPI schema.
    """
    
    def __init__(self, lidar_rtx):
        """
        Args:
            lidar_rtx: LidarRtx instance wrapping the OmniLidar prim
        """
        self._lidar = lidar_rtx
        self.point_cloud = Buffer(tags=["point_cloud"])
        self.intensities = Buffer(tags=["intensities"])
        self.distances = Buffer(tags=["distances"])
        self._debug_draw_enabled = False
        self._debug_draw_color = (0.0, 1.0, 0.0, 1.0)
        self._draw_interface = None

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        config_file_name: str = "Example_Rotary",
        **sensor_attributes,
    ) -> "LidarSensor":
        """
        Creates an RTX Lidar using the LidarRtx class.
        
        Args:
            prim_path: Path where the OmniLidar prim will be created
            translation: (x, y, z) position
            orientation: (w, x, y, z) quaternion orientation
            config_file_name: Name of the lidar config (e.g., "Example_Rotary", "HESAI_XT32_SD10")
            **sensor_attributes: Additional attributes for the OmniLidar prim
        
        Returns:
            LidarSensor instance
        """
        from isaacsim.sensors.rtx import LidarRtx
        
        print(f"[LidarSensor] Building at {prim_path} with config {config_file_name}")
        
        # Use LidarRtx directly - it will create the OmniLidar prim internally
        lidar = LidarRtx(
            prim_path=prim_path,
            translation=np.array(translation),
            orientation=np.array(orientation),
            config_file_name=config_file_name,
        )
        
        return cls(lidar)
    
    @classmethod
    def attach(cls, prim_path: str) -> "LidarSensor":
        """
        Attach to an existing OmniLidar prim.
        """
        from isaacsim.sensors.rtx import LidarRtx
        
        lidar = LidarRtx(prim_path=prim_path)
        try:
            lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
        except Exception as e:
            print(f"[LidarSensor] Warning: Could not attach annotator: {e}")
        
        return cls(lidar)

#Funçoes para o lidar sensor ==========================================
    def update_state(self):
        """
        Collects data from the RTX Lidar annotators.
        Implements fallback: if RTX Lidar fails or returns empty, try replicator annotator 'point_cloud'.
        """
        point_cloud_set = False
        try:
            # RTX Lidar annotators
            if hasattr(self._lidar, '_annotators') and self._lidar._annotators:
                for name, annotator in self._lidar._annotators.items():
                    data = annotator.get_data()
                    if data is not None and 'data' in data:
                        pc = data['data']
                        if pc is not None and hasattr(pc, 'shape') and pc.shape[0] > 0:
                            self.point_cloud.set_value(pc)
                            point_cloud_set = True
                        if 'intensity' in data and data['intensity'].shape[0] > 0:
                            self.intensities.set_value(data['intensity'])
                        if 'distance' in data and data['distance'].shape[0] > 0:
                            self.distances.set_value(data['distance'])
                        break
        except Exception as e:
            print(f"[LidarSensor] RTX Lidar annotator error: {e}")

        # Fallback: replicator annotator 'pointcloud'
        if not point_cloud_set:
            try:
                import omni.replicator.core as rep
                # Cria render product se necessário
                if not hasattr(self, '_render_product') or self._render_product is None:
                    self._render_product = rep.create.render_product(self._lidar.prim_path, (512, 512), force_new=True)
                annotator = rep.AnnotatorRegistry.get_annotator("pointcloud")
                if annotator is not None:
                    annotator.attach(self._render_product)
                    pc = annotator.get_data()
                    if pc is not None and hasattr(pc, 'shape') and pc.shape[0] > 0:
                        self.point_cloud.set_value(pc)
                        print("[LidarSensor] Fallback: pointcloud from replicator annotator.")
            except Exception as e:
                print(f"[LidarSensor] Fallback failed: {e}")

        super().update_state()
    
    def enable_rendering(self):
        """Enable lidar rendering."""
        try:
            self._lidar.resume()
        except:
            pass
    
    def disable_rendering(self):
        #Disable lidar rendering.
        try:
            self._lidar.pause()
        except:
            pass

    # Métodos de compatibilidade para replay (Lidar não usa esses annotators)
    def enable_rgb_rendering(self):
        """Lidar não tem RGB - método vazio para compatibilidade."""
        pass
    
    def enable_segmentation_rendering(self):
        """Lidar não tem segmentação - método vazio para compatibilidade."""
        pass
    
    def enable_depth_rendering(self):
        """Lidar não tem depth image - método vazio para compatibilidade."""
        pass
    
    def enable_instance_id_segmentation_rendering(self):
        """Lidar não tem instance segmentation - método vazio para compatibilidade."""
        pass
    
    def enable_normals_rendering(self):
        """Lidar não tem normals - método vazio para compatibilidade."""
        pass

"""
    def enable_debug_draw(self, color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)):
        #Habilita a visualização dos pontos do Lidar no viewport usando debug draw.
        
        #Args:
            #color: Cor RGBA dos pontos (default: verde)

        self._debug_draw_enabled = True
        self._debug_draw_color = color
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._draw_interface = _debug_draw.acquire_debug_draw_interface()
            print(f"[LidarSensor] Debug draw enabled with color {color}")
        except ImportError as e:
            print(f"[LidarSensor] Warning: debug_draw not available: {e}")
            self._debug_draw_enabled = False
    
    def disable_debug_draw(self):
        #Desabilita a visualização debug draw.
        self._debug_draw_enabled = False
        if self._draw_interface is not None:
            try:
                self._draw_interface.clear_points()
            except:
                pass
    """


"""
    def draw_point_cloud(self, point_size: float = 2.0, subsample: int = 10, draw_every_n_frames: int = 2):
    
        #Desenha os pontos do Lidar no viewport.
        #Chamar após update_state() no loop de física.
        
        #Args:
            #point_size: Tamanho dos pontos
            #subsample: Pegar 1 a cada N pontos (default: 20 = ~3500 pontos de 70000)
            #draw_every_n_frames: Desenhar a cada N frames (default: 5)
        
        if not self._debug_draw_enabled or self._draw_interface is None:
            return
        
        # Controle de frequência
        if not hasattr(self, '_draw_frame_count'):
            self._draw_frame_count = 0
        self._draw_frame_count += 1
        
        if self._draw_frame_count % draw_every_n_frames != 0:
            return
            
        points = self.point_cloud.get_value()
        if points is None:
            return
            
        # Converter para numpy array se necessário
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if points.size == 0:
            return
        
        # Garantir shape correto (N, 3)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        
        # Subsample para performance
        if subsample > 1:
            points = points[::subsample]
        
        num_points = len(points)
        if num_points == 0:
            return
        
        # Limpar pontos anteriores e desenhar novos
        try:
            self._draw_interface.clear_points()
            colors = [self._debug_draw_color] * num_points
            sizes = [point_size] * num_points
            self._draw_interface.draw_points(points.tolist(), colors, sizes)
        except:
            pass
        """