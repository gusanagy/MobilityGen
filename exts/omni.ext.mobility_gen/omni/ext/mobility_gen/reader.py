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


import PIL.Image
import glob
import numpy as np
import os
from collections import OrderedDict


from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.config import Config


class Reader:

    def __init__(self, recording_path: str):
        self.recording_path = recording_path
        
        state_dict_paths = glob.glob(os.path.join(
            self.recording_path, "state", "common", "*.npy"
        ))

        steps = [int(os.path.basename(path).split('.')[0]) for path in state_dict_paths]
        self.steps = sorted(steps)

        self.rgb_folders = glob.glob(os.path.join(self.recording_path, "state", "rgb", "*"))
        self.segmentation_folders = glob.glob(os.path.join(self.recording_path, "state", "segmentation", "*"))
        self.depth_folders = glob.glob(os.path.join(self.recording_path, "state", "depth", "*"))
        self.normals_folders = glob.glob(os.path.join(self.recording_path, "state", "normals", "*"))
        
        self.point_cloud_folders = glob.glob(os.path.join(self.recording_path, "state", "point_cloud", "*"))
        self.point_cloud_names = [os.path.basename(folder) for folder in self.point_cloud_folders]

        self.fisheye_folders = glob.glob(os.path.join(self.recording_path, "state", "fisheye", "*"))
        self.fisheye_names = [os.path.basename(folder) for folder in self.fisheye_folders]
        
        self.rgb_names = [os.path.basename(folder) for folder in self.rgb_folders]
        self.segmentation_names = [os.path.basename(folder) for folder in self.segmentation_folders]
        self.depth_names = [os.path.basename(folder) for folder in self.depth_folders]
        self.normals_names = [os.path.basename(folder) for folder in self.normals_folders]

    def read_config(self) -> Config:
        with open(os.path.join(self.recording_path, "config.json"), 'r') as f:
            config = Config.from_json(f.read())
        return config

    def read_occupancy_map(self):
        return OccupancyMap.from_ros_yaml(os.path.join(self.recording_path, "occupancy_map", "map.yaml"))
    
    def read_rgb(self, name: str, index: int):
        step = self.steps[index]
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "rgb", name, f"{step:08d}.jpg"))
        return np.asarray(image)
    
    def read_state_dict_rgb(self, index: int):
        rgb_dict = OrderedDict()
        for name in self.rgb_names:
            data = self.read_rgb(name, index)
            rgb_dict[name] = data
        return rgb_dict
    
    def read_segmentation(self, name: str, index: int):
        step = self.steps[index]
        npy_path = os.path.join(self.recording_path, "state", "segmentation", name, f"{step:08d}.npy")
        if os.path.exists(npy_path):
            return np.load(npy_path)
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "segmentation", name, f"{step:08d}.png"))
        return np.asarray(image)
    
    def read_normals(self, name: str, index: int):
        step = self.steps[index]
        data = np.load(
            os.path.join(self.recording_path, "state", "normals", name, f"{step:08d}.npy")
        )
        return data
    
    def read_lidar(self, name: str, index: int):
        step = self.steps[index]
        npy_path = os.path.join(self.recording_path, "state", "lidar", name, f"{step:08d}.npy")
        if os.path.exists(npy_path):
            point_cloud = np.load(npy_path)
            return point_cloud
        else:
            return None

    def read_state_dict_segmentation(self, index: int):
        segmentation_dict = OrderedDict()
        for name in self.segmentation_names:
            data = self.read_segmentation(name, index)
            segmentation_dict[name] = data
        return segmentation_dict
    
    def read_depth(self, name: str, index: int, eps=1e-6):
        step = self.steps[index]
        npy_path = os.path.join(self.recording_path, "state", "depth", name, f"{step:08d}.npy")
        if os.path.exists(npy_path):
            inverse_depth = np.load(npy_path).astype(np.float32)
            depth = 65535 / (inverse_depth + eps) - 1.0
            return depth
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "depth", name, f"{step:08d}.png")).convert("I;16")
        depth = 65535 / (np.asarray(image).astype(np.float32) + eps) - 1.0
        return depth
    
    def read_state_dict_depth(self, index: int):
        depth_dict = OrderedDict()
        for name in self.depth_names:
            data = self.read_depth(name, index)
            depth_dict[name] = data
        return depth_dict

    def read_state_dict_normals(self, index: int):
        normals_dict = OrderedDict()
        for name in self.normals_names:
            data = self.read_normals(name, index)
            normals_dict[name] = data
        return normals_dict

    def read_state_dict_common(self, index: int):
        step = self.steps[index]
        state_dict = np.load(os.path.join(self.recording_path, "state", "common", f"{step:08d}.npy"), allow_pickle=True).item()
        return state_dict

    def read_fisheye(self, name: str, index: int):
        step = self.steps[index]
        image_path = os.path.join(self.recording_path, "state", "fisheye", name, f"{step:08d}.jpg")
        if os.path.exists(image_path):
            image = PIL.Image.open(image_path)
            return np.asarray(image)
        else:
            return None

    def read_lidar(self, name: str, index: int):
        # Mantém compatibilidade, mas lê de point_cloud
        return self.read_point_cloud(name, index)

    def read_state_dict_point_cloud(self, index: int):
        """Lê os dados de point cloud do lidar salvos como .npy"""
        point_cloud_dict = OrderedDict()
        step = self.steps[index]
        for name in self.point_cloud_names:
            npy_path = os.path.join(self.recording_path, "state", "point_cloud", name, f"{step:08d}.npy")
            if os.path.exists(npy_path):
                point_cloud_dict[name] = np.load(npy_path)
        return point_cloud_dict
    
    def read_state_dict_fisheye(self, index: int):
        """Lê os dados de imagens fisheye"""
        fisheye_dict = OrderedDict()
        step = self.steps[index]
        for name in self.fisheye_names:
            image_path = os.path.join(self.recording_path, "state", "fisheye", name, f"{step:08d}.jpg")
            if os.path.exists(image_path):
                image = PIL.Image.open(image_path)
                fisheye_dict[name] = np.asarray(image)
        return fisheye_dict

    def read_state_dict(self, index: int):

        state_dict = self.read_state_dict_common(index)
        rgb_dict = self.read_state_dict_rgb(index)
        segmentation_dict = self.read_state_dict_segmentation(index)
        depth_dict = self.read_state_dict_depth(index)
        normals_dict = self.read_state_dict_normals(index)
        point_cloud_dict = self.read_state_dict_point_cloud(index)
        fisheye_dict = self.read_state_dict_fisheye(index)

        full_dict = OrderedDict()
        full_dict.update(state_dict)
        full_dict.update(rgb_dict)
        full_dict.update(segmentation_dict)
        full_dict.update(depth_dict)
        full_dict.update(normals_dict)
        full_dict.update(point_cloud_dict)
        if full_dict.get("point_cloud") is None:
            print(f"Warning: No point cloud data found for step {index}. 'point_cloud' key will be set to None.")
            full_dict["point_cloud"] = None
        full_dict.update(fisheye_dict)

        return full_dict
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, index: int):
        return self.read_state_dict(index)