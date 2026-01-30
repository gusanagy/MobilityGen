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

# tenho que editar os sensores de camera usando esse modulo como base 
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
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._rgb_annotator.attach(self._render_product)

    def enable_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._segmentation_annotator is not None:
            return
        self._segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params=dict(colorize=False)
        )
        self._segmentation_annotator.attach(self._render_product)

    def enable_instance_id_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._instance_id_segmentation_annotator is not None:
            return
        self._instance_id_segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params=dict(colorize=False)
        )
        self._instance_id_segmentation_annotator.attach(self._render_product)

    def enable_depth_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._depth_annotator is not None:
            return
        self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_camera"
        )
        self._depth_annotator.attach(self._render_product)

    def enable_normals_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._normals_annotator is not None:
            return
        self._normals_annotator = rep.AnnotatorRegistry.get_annotator(
            "normals"
        )
        self._normals_annotator.attach(self._render_product)

    def update_state(self):
        if self._rgb_annotator is not None:
            self.rgb_image.set_value(
                self._rgb_annotator.get_data()[:, :, :3]
            )
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
# RealSense em formato HawkCamera
# ================================================
class RealSenseRGBDCamera(Sensor):
    """
    Câmera “tipo RealSense” (RGB-D) com o mesmo padrão estrutural de HawkCamera:
    - Atributos de classe: usd_url, resolution, camera_path (subprim interno)
    - build(): injeta a referência USD e delega para attach()
    - attach(): empacota o prim interno em uma Camera (sua classe)
    """
    # Ajuste para o caminho real do asset no seu servidor/Omniverse
    usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/NVIDIA/Kaya/props/realsense.usd"
    resolution: Tuple[int, int] = (1280, 720)
    camera_path: str = "camera"  # subcaminho do prim de câmera dentro do USD referenciado

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(cls, prim_path: str) -> "RealSenseRGBDCamera":
        """
        Cria um Xform em `prim_path`, adiciona a referência para o USD da RealSense
        e retorna o wrapper com a Camera interna.
        """
        stage = get_stage()

        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url,
        )

        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "RealSenseRGBDCamera":
        """
        Resolve o caminho absoluto do prim interno da câmera (ex.: "<prim_path>/camera")
        e o empacota em uma instância da sua classe Camera, com a resolução padrão.
        """
        cam_full_path = os.path.join(prim_path, cls.camera_path)
        cam = Camera(cam_full_path, cls.resolution)
        return RealSenseRGBDCamera(cam)


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
# CÂMERAS BEV (padrão HawkCamera; sem atalhos próprios)
# =========================================================

class BevTopDownCamera(Sensor):
    """
    Câmera ortográfica (vista superior métrica) – ideal para GT BEV.
    build(): cria/configura o prim ortográfico (apertures em mm) e posiciona em +Z.
    attach(): envolve o prim em uma Camera (sua classe).
    """
    usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd" 
    resolution: Tuple[int, int] = (1024, 1024)

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        height_m: float = 5.0,
        view_width_m: float = 14.0,
        view_height_m: float = 14.0,
        resolution: Tuple[int, int] = (1024, 1024),
        near: float = 0.05,
        far: float = 2000.0,
    ) -> "BevTopDownCamera":
        """
        - horizontal/verticalAperture em **milímetros** (USD).
        - Para cobrir exatamente `view_width_m` x `view_height_m` em mundo, use mm = metros * 1000.
        - Em ortográfica, a câmera olha por padrão para -Z; subir em +Z basta.
        """
        cam_prim = _define_camera_prim(prim_path)
        cam_prim.CreateProjectionAttr(UsdGeom.Tokens.orthographic)

        # ✅ Conversão correta: metros -> milímetros
        cam_prim.CreateHorizontalApertureAttr(view_width_m * 1000.0)
        cam_prim.CreateVerticalApertureAttr(view_height_m * 1000.0)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))

        # Posição: acima do plano (olhando -Z por padrão)
        _xform_translate(prim_path, (0.0, 0.0, float(height_m)))

        return cls.attach(prim_path, resolution)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "BevTopDownCamera":
        res = resolution if resolution is not None else cls.resolution
        return BevTopDownCamera(Camera(prim_path, res))


class BevFrontDownCamera(Sensor):
    """
    Câmera perspectiva à frente e inclinada para baixo.
    - FOV horizontal definido por hfov_deg.
    - verticalAperture calculada pelo aspect ratio da resolução (robusto para não-16:9).
    """
    usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd"
    resolution: Tuple[int, int] = (1280, 720)

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        forward_m: float = 0.8,
        height_m: float = 1.8,
        pitch_down_deg: float = 55.0,
        hfov_deg: float = 70.0,
        resolution: Tuple[int, int] = (1280, 720),
        near: float = 0.05,
        far: float = 1000.0,
        filmback_mm: float = 36.0,  # horizontalAperture "clássica" de 36mm
    ) -> "BevFrontDownCamera":
        """
        Constrói uma câmera perspectiva com HFOV alvo:
        - focal = 0.5 * horiz_ap_mm / tan(HFOV/2)
        - verticalAperture = horiz_ap_mm / aspect (para respeitar a resolução escolhida)
        """
        
        cam_prim = _define_camera_prim(prim_path)
        cam_prim.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))

        # Óptica a partir do HFOV
        horiz_ap_mm = float(filmback_mm)
        focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)

        # Usa o aspect da resolução (robusto para qualquer res)
        res_x, res_y = resolution
        aspect = (res_x / res_y) if (res_x and res_y) else (16.0 / 9.0)
        vert_ap_mm = horiz_ap_mm / aspect

        cam_prim.CreateHorizontalApertureAttr(horiz_ap_mm)
        cam_prim.CreateVerticalApertureAttr(vert_ap_mm)
        cam_prim.CreateFocalLengthAttr(focal_mm)

        # Pose: desloca à frente e inclina "olhando para baixo"
        _xform_translate(prim_path, (float(forward_m), 0.0, float(height_m)))
        qw, qx, qy, qz = _quat_from_euler_xyz(0.0, -float(pitch_down_deg), 0.0)
        _xform_orient_quat(prim_path, (qw, qx, qy, qz))

        return cls.attach(prim_path, resolution)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "BevFrontDownCamera":
        res = resolution if resolution is not None else cls.resolution
        return BevFrontDownCamera(Camera(prim_path, res))



# =========================================================
# Camera nuScenes
# =========================================================

class NuScenesCamera(Sensor):
    """
    Single perspective camera using the SG8S-AR0820C-5300-G2A-H60SA USD asset.
    Works as a drop-in 'front_camera_type' for Robot.build_front_camera().
    """
    usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Sensors/Sensing/SG8/H60SA/SG8S-AR0820C-5300-G2A-H60SA.usd"
    resolution: Tuple[int, int] = (1920, 1200)
    # The camera prim path inside the USD asset
    camera_subprim: str = "SG8S_AR0820C_5300_G2A_H60SA_01"

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        resolution: Tuple[int, int] = None,
    ) -> "NuScenesCamera":
        """
        Loads the SG8S camera USD asset at prim_path and wraps the internal camera prim.
        """
        stage = get_stage()
        
        # Add the USD reference at prim_path (like HawkCamera does)
        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url
        )
        
        return cls.attach(prim_path, resolution)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "NuScenesCamera":
        res = resolution or cls.resolution
        # Full path to camera inside the USD: prim_path/camera_subprim
        camera_full_path = os.path.join(prim_path, cls.camera_subprim)

        return NuScenesCamera(Camera(camera_full_path, res))
#========================================================


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

    def __init__(self, cam: Camera):
        self.cam = cam

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
        res = resolution or cls.resolution
        return FisheyeCamera(Camera(prim_path, res))