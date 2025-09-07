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
import PIL.Image
import numpy as np
import shutil

from omni.ext.mobility_gen.utils.global_utils import save_stage
from omni.ext.mobility_gen.config import Config
from omni.ext.mobility_gen.occupancy_map import OccupancyMap


class Writer:

    def __init__(self, path: str):
        self.path = path

    def write_state_dict_common(self, state_dict: dict, step: int):
        dict_folder = os.path.join(self.path, "state", "common")
        if not os.path.exists(dict_folder):
            os.makedirs(dict_folder)
        state_dict_path = os.path.join(dict_folder, f"{step:08d}.npy")
        np.save(state_dict_path, state_dict)

    def write_state_dict_rgb(self, state_rgb: dict, step: int):
        for name, value in state_rgb.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "rgb", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.jpg")
                image = PIL.Image.fromarray(value)
                image.save(image_path)

    def write_state_dict_segmentation(self, state_segmentation: dict, step: int):
        for name, value in state_segmentation.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "segmentation", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.png")
                image = PIL.Image.fromarray(value)
                image.save(image_path)

    def write_state_dict_depth(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "depth", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Inverse depth 16bit
                inverse_depth = 1.0 / (1.0 + value)
                inverse_depth = (65535 * inverse_depth).astype(np.uint16)
                image = PIL.Image.fromarray(inverse_depth, "I;16")
                
                output_path = os.path.join(output_folder, f"{step:08d}.png")

                image.save(output_path)

    def write_state_dict_normals(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "normals", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, f"{step:08d}.npy")
                np.save(output_path, value)

######## writer bev ################################################
    def write_bev(self, state_bev:dict, step:int):
        for names, value in state_bev.items():
            if value is not None:
                bev_folder = os.path.join(self.path, "state", "bev", names)
                os.makedirs(bev_folder, exist_ok=True)
                
                bev_path = os.path.join(bev_folder, f"{step:08d}.png")
                bev = PIL.image.fromarray(value)
                bev.save(bev_path)
####################################################################
    def write_stage(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        save_stage(
            os.path.join(self.path, "stage.usd")
        )

    def copy_stage(self, input_path: str):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        shutil.copyfile(input_path, os.path.join(self.path, "stage.usd"))

    def write_config(self, config: Config):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(os.path.join(self.path, "config.json"), 'w') as f:
            f.write(config.to_json())

    def write_occupancy_map(self, occupancy_map: OccupancyMap):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        occupancy_map.save_ros(os.path.join(self.path, "occupancy_map"))

    def copy_init(self, other_path: str):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        shutil.copyfile(os.path.join(other_path, "stage.usd"), os.path.join(self.path, "stage.usd"))
        shutil.copyfile(os.path.join(other_path, "config.json"), os.path.join(self.path, "config.json"))
        shutil.copytree(os.path.join(other_path, "occupancy_map"), os.path.join(self.path, "occupancy_map"))
    


    #writer nuscenes mexer depois
    def write_nuscenes(self, state_nuscenes:dict, step:int):    
        for names, value in state_nuscenes.items():
            if value is not None:
                nuscenes_folder = os.path.join(self.path, "state", "nuScenes", names)
                os.makedirs(nuscenes_folder, exist_ok=True)
                
                bev_path = os.path.join(nuscenes_folder, f"{step:08d}.png")
                bev = PIL.image.fromarray(value)
                bev.save(bev_path)
        pass
    #fisheye
    def write_fisheye(self, state_fisheye:dict, step:int):
        pass
    #lidar
    #mui
