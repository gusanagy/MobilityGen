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


import asyncio
import numpy as np
import os
import datetime
import tempfile
import glob

import omni.ext
import omni.ui as ui

from omni.ext.mobility_gen.utils.global_utils import save_stage
from omni.ext.mobility_gen.writer import Writer
from omni.ext.mobility_gen.inputs import GamepadDriver, KeyboardDriver
from omni.ext.mobility_gen.scenarios import SCENARIOS, Scenario
from omni.ext.mobility_gen.utils.global_utils import get_world
from omni.ext.mobility_gen.utils.global_utils import get_stage
from omni.ext.mobility_gen.utils.path_utils import PathHelper
from pxr import UsdGeom, Usd, Gf
from omni.ext.mobility_gen.robots import ROBOTS
from omni.ext.mobility_gen.config import Config
from omni.ext.mobility_gen.build import build_scenario_from_config


if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ['MOBILITY_GEN_DATA']
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")

RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
SCENARIOS_DIR = os.path.join(DATA_DIR, "scenarios")


class MobilityGenExtension(omni.ext.IExt):
    """
    MobilityGenExtension: Extension UI and runtime hooks.

    Responsibilities:
      - provide a small UI to build a scenario from a USD world, robot and scenario type
      - manage recording and pointcloud writer lifecycle
      - expose quick sensor controls (enable/disable cameras/lidar)
      - coordinate planning requests (Plan & Start Auto) via robot or scenario
    """

    def on_startup(self, ext_id):

        # Input drivers
        self.keyboard = KeyboardDriver.connect()
        self.gamepad = GamepadDriver.connect()

        # Runtime state
        self.scenario: Scenario = None
        self.config: Config = None

        self.count = 0

        self.scenario_path: str | None = None
        self.cached_stage_path: str | None = None

        self.writer: Writer | None = None
        self.step: int = 0
        self.is_recording: bool = False
        self.recording_enabled: bool = False
        self.recording_time: float = 0.

        # Image provider for occupancy map preview
        self._occupancy_map_image_provider = omni.ui.ByteImageProvider()

        # Visualization window for occupancy map
        self._visualize_window = omni.ui.Window("MobilityGen - Occupancy Map", width=300, height=300)
        with self._visualize_window.frame:
            self._occ_map_frame = ui.Frame()
            self._occ_map_frame.set_build_fn(self.build_occ_map_frame)

        # discover available USD worlds in the working directory and DATA_DIR
        try:
            self._available_worlds = self._scan_worlds()
        except Exception:
            self._available_worlds = []

        self._teleop_window = omni.ui.Window("MobilityGen", width=300, height=300)

        with self._teleop_window.frame:
            with ui.VStack():
                with ui.VStack():
                    with ui.HStack():
                        ui.Label("USD Path / URL")
                        self.scene_usd_field_string_model = ui.SimpleStringModel()
                        self.scene_usd_field = ui.StringField(model=self.scene_usd_field_string_model, height=25)

                    with ui.HStack():
                        ui.Label("Scenario Type")
                        self.scenario_combo_box = ui.ComboBox(0, *SCENARIOS.names())

                    with ui.HStack():
                        ui.Label("Robot Type")
                        self.robot_combo_box = ui.ComboBox(0, *ROBOTS.names())
                
                    # -- Build button --
                    ui.Button("Build", clicked_fn=self.build_scenario)

                    # -- Quick parameter models (used by recording and buttons) --
                    # These models exist even if the full recording UI is not shown,
                    # so the background logic in on_physics can reference them.
                    try:
                        self.record_pointclouds_model = ui.SimpleBoolModel(False)
                    except Exception:
                        self.record_pointclouds_model = None
                    try:
                        self.pointcloud_interval_model = ui.SimpleIntModel(1)
                    except Exception:
                        self.pointcloud_interval_model = None
                    try:
                        self.annotate_bboxes_model = ui.SimpleBoolModel(False)
                    except Exception:
                        self.annotate_bboxes_model = None
                    # fallback internal format/index for environments where the
                    # pointcloud format ComboBox is not present in the UI
                    self._pc_format_items = ["npy", "ply", "pcd"]
                    self._pc_format_index = 0

                    # -- Quick Params frame: organized controls after Build --
                    with ui.Frame():
                        ui.Label("Quick Params")
                        with ui.HStack(height=40):
                            # Left column: Save stage and occupancy map visibility
                            with ui.VStack(width=180, spacing=4):
                                ui.Button("Save Stage", clicked_fn=self._save_stage_now)
                                # Show occupancy map checkbox + label on same row
                                try:
                                    if not hasattr(self, '_show_occ_map_model') or self._show_occ_map_model is None:
                                        self._show_occ_map_model = ui.SimpleBoolModel(False)
                                except Exception:
                                    self._show_occ_map_model = None
                                with ui.HStack():
                                    if self._show_occ_map_model is not None:
                                        cb = ui.CheckBox(model=self._show_occ_map_model)
                                        # Deterministic: set visible when checked, hide when unchecked
                                        def _on_show_occ_change(value=None):
                                            try:
                                                val = self._show_occ_map_model.get_value()
                                                # ensure the toggle sets to a deterministic visible state
                                                if val:
                                                    try:
                                                        self._visualize_window.set_visible(True)
                                                    except Exception:
                                                        try:
                                                            self._visualize_window.visible = True
                                                        except Exception:
                                                            pass
                                                else:
                                                    try:
                                                        self._visualize_window.set_visible(False)
                                                    except Exception:
                                                        try:
                                                            self._visualize_window.visible = False
                                                        except Exception:
                                                            pass
                                            except Exception:
                                                pass
                                        try:
                                            cb.model.add_value_changed_fn(_on_show_occ_change)
                                        except Exception:
                                            pass
                                    ui.Label("Show Occ Map")

                            # Middle column: recording toggles
                            with ui.VStack(width=200, spacing=4):
                                # Record PointClouds checkbox and label
                                try:
                                    if self.record_pointclouds_model is None:
                                        self.record_pointclouds_model = ui.SimpleBoolModel(False)
                                except Exception:
                                    self.record_pointclouds_model = None
                                with ui.HStack():
                                    if self.record_pointclouds_model is not None:
                                        cb_rec = ui.CheckBox(model=self.record_pointclouds_model)
                                    ui.Label("Record PointClouds")

                                # Annotate BBoxes checkbox
                                try:
                                    if self.annotate_bboxes_model is None:
                                        self.annotate_bboxes_model = ui.SimpleBoolModel(False)
                                except Exception:
                                    self.annotate_bboxes_model = None
                                with ui.HStack():
                                    if self.annotate_bboxes_model is not None:
                                        cb_ann = ui.CheckBox(model=self.annotate_bboxes_model)
                                    ui.Label("Annotate BBoxes")

                            # Right column: format and interval selectors
                            with ui.VStack(width=220, spacing=4):
                                with ui.HStack():
                                    ui.Label("Format")
                                    try:
                                        items = self._pc_format_items
                                        self._pc_format_combo = ui.ComboBox(self._pc_format_index, *items)
                                        try:
                                            self._pc_format_combo.model.add_value_changed_fn(lambda: setattr(self, '_pc_format_index', self._pc_format_combo.model.get_item_value_model().get_value_as_int()))
                                        except Exception:
                                            pass
                                    except Exception:
                                        # fallback label if combo fails
                                        ui.Label(self._pc_format_items[self._pc_format_index] if hasattr(self, '_pc_format_items') else "npy")

                                with ui.HStack():
                                    ui.Label("Interval (frames)")
                                    try:
                                        self._interval_combo = ui.ComboBox(0, "1", "5")
                                        try:
                                            self._interval_combo.model.add_value_changed_fn(lambda: (self.pointcloud_interval_model.set_value(int(["1","5"][self._interval_combo.model.get_item_value_model().get_value_as_int()])) if self.pointcloud_interval_model is not None else None))
                                        except Exception:
                                            pass
                                    except Exception:
                                        ui.Label(str(self.pointcloud_interval_model.get_value() if self.pointcloud_interval_model is not None else "1"))

                with ui.VStack():
                    self.recording_count_label = ui.Label("")
                    self.recording_dir_label = ui.Label(f"Output directory: {RECORDINGS_DIR}")
                    self.recording_name_label = ui.Label("")
                    self.recording_step_label = ui.Label("")

                    ui.Button("Reset", clicked_fn=self.reset)
                    with ui.HStack():
                        ui.Button("Start Recording", clicked_fn=self.enable_recording)
                        ui.Button("Stop Recording", clicked_fn=self.disable_recording)

        self.update_recording_count()
        self.clear_recording()

    def build_occ_map_frame(self):
        if self.scenario is not None:
            with ui.VStack():
                image_widget = ui.ImageWithProvider(
                    self._occupancy_map_image_provider
                )

    def draw_occ_map(self):
        if self.scenario is not None:
            image = self.scenario.occupancy_map.ros_image().copy().convert("RGBA")
            data = list(image.tobytes())
            self._occupancy_map_image_provider.set_bytes_data(data, [image.width, image.height])
            self._occ_map_frame.rebuild()


    def update_recording_count(self):
        num_recordings = len(glob.glob(os.path.join(RECORDINGS_DIR, "*")))
        self.recording_count_label.text = f"Number of recordings: {num_recordings}"

    # ------- UI -> Config helper -------
    def create_config(self):
        # Minimal, clear create_config: read UI values and return Config
        try:
            scenario_type = list(SCENARIOS.names())[self.scenario_combo_box.model.get_item_value_model().get_value_as_int()]
        except Exception:
            scenario_type = list(SCENARIOS.names())[0]
        try:
            robot_type = list(ROBOTS.names())[self.robot_combo_box.model.get_item_value_model().get_value_as_int()]
        except Exception:
            robot_type = list(ROBOTS.names())[0]

        # try:
        #     scene_path = self.scene_usd_field_string_model.as_string
        # except Exception:
        scene_path = "/home/pdi_4/Documents/Documentos/bevlog-isaac/mundo_pallets.usd"

        config = Config(
            scenario_type=scenario_type,
            robot_type=robot_type,
            scene_usd=scene_path,
        )
        return config
    
    def scenario_type(self):
        index = self.scenario_combo_box.model.get_item_value_model().get_value_as_int()
        return SCENARIOS.get_index(index)
    
    def on_shutdown(self):
        # Defensive shutdown: some Kit shutdown sequences remove the world
        # before extensions are asked to shutdown, so `get_world()` may
        # return None. Guard all disconnect/remove calls to avoid raising
        # during extension shutdown.
        try:
            if hasattr(self, 'keyboard') and self.keyboard is not None:
                try:
                    self.keyboard.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, 'gamepad') and self.gamepad is not None:
                try:
                    self.gamepad.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            world = get_world()
            if world is not None:
                try:
                    world.remove_physics_callback("scenario_physics", self.on_physics)
                except Exception:
                    pass
        except Exception:
            pass

    def start_new_recording(self):
        recording_name = datetime.datetime.now().isoformat()
        recording_path = os.path.join(RECORDINGS_DIR, recording_name)
        writer = Writer(recording_path)
        writer.write_config(self.config)
        writer.write_occupancy_map(self.scenario.occupancy_map)
        writer.copy_stage(self.cached_stage_path)
        self.step = 0
        self.recording_time = 0.
        self.recording_name_label.text = f"Current recording name: {recording_name}"
        self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"
        self.writer = writer
        self.update_recording_count()
    
    def clear_recording(self):
        self.writer = None
        self.recording_name_label.text = "Current recording name: "
        self.recording_step_label.text = "Current recording duration: "

    def clear_scenario(self):
        self.scenario = None
        self.cached_stage_path = None

    def enable_recording(self):
        if not self.recording_enabled:
            if self.scenario is not None:
                self.start_new_recording()
            self.recording_enabled = True

    def disable_recording(self):
        self.recording_enabled = False
        self.clear_recording()

    def reset(self):
        self.writer = None
        self.scenario.reset()
        if self.recording_enabled:
            self.start_new_recording()

    def on_physics(self, step_size: int):

        if self.scenario is not None:

            is_alive = self.scenario.step(step_size)

            if not is_alive:
                self.reset()
            
            if self.writer is not None:
                state_dict = self.scenario.state_dict_common()
                self.writer.write_state_dict_common(state_dict, step=self.step)
                # Pointcloud auto-recording
                try:
                    record_pc = self.record_pointclouds_model.get_value()
                except Exception:
                    record_pc = False

                if record_pc:
                    try:
                        interval = int(self.pointcloud_interval_model.get_value())
                    except Exception:
                        interval = 1
                    if interval <= 0:
                        interval = 1
                    if (self.step % interval) == 0:
                        state_pc = self.scenario.state_dict_pointcloud()
                        fmt = "npy"
                        try:
                            fmt = ["npy", "ply", "pcd"][self.pointcloud_format_combo.model.get_item_value_model().get_value_as_int()]
                        except Exception:
                            # fallback to internal index if the UI combo is not present
                            try:
                                fmt = self._pc_format_items[self._pc_format_index]
                            except Exception:
                                fmt = "npy"
                        try:
                            self.writer.write_state_dict_pointcloud(state_pc, step=self.step, save_format=fmt)
                        except Exception:
                            # Best-effort: continue if writer fails
                            pass

                        # Persist per-sensor metadata (pose) next to the pointcloud
                        try:
                            metadata = {}
                            modules = self.scenario.named_modules()
                            for full_name, arr_value in state_pc.items():
                                # sensor module name is the prefix before the last '.':
                                if "." in full_name:
                                    module_name = full_name.rsplit('.', 1)[0]
                                else:
                                    module_name = full_name
                                module = modules.get(module_name, None)
                                if module is None:
                                    continue
                                pos = None
                                ori = None
                                try:
                                    if hasattr(module, 'position') and module.position.get_value() is not None:
                                        pos = module.position.get_value()
                                except Exception:
                                    pos = None
                                try:
                                    if hasattr(module, 'orientation') and module.orientation.get_value() is not None:
                                        ori = module.orientation.get_value()
                                except Exception:
                                    ori = None

                                # fallback: try _xform_prim
                                if (pos is None or ori is None) and hasattr(module, '_xform_prim'):
                                    try:
                                        p, o = module._xform_prim.get_world_pose()
                                        if pos is None:
                                            pos = p
                                        if ori is None:
                                            ori = o
                                    except Exception:
                                        pass

                                # detect pointcloud fields based on array shape
                                fields = None
                                try:
                                    if arr_value is not None:
                                        a = np.asarray(arr_value)
                                        if a.ndim == 2:
                                            ncol = a.shape[1]
                                            if ncol == 3:
                                                fields = ["x", "y", "z"]
                                            elif ncol == 4:
                                                fields = ["x", "y", "z", "intensity"]
                                            elif ncol == 6:
                                                fields = ["x", "y", "z", "r", "g", "b"]
                                            elif ncol == 7:
                                                fields = ["x", "y", "z", "r", "g", "b", "intensity"]
                                            else:
                                                fields = ["x", "y", "z"]
                                except Exception:
                                    fields = None

                                if pos is not None or ori is not None or fields is not None:
                                    metadata[module_name] = {
                                        'position': None if pos is None else [float(x) for x in list(pos)],
                                        'orientation': None if ori is None else [float(x) for x in list(ori)],
                                        'prim_path': getattr(module, '_prim_path', None),
                                        'fields': fields
                                    }
                            if len(metadata) > 0:
                                try:
                                    self.writer.write_pointcloud_metadata(metadata, step=self.step)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Annotations (2D/3D) using prim bounds if requested
                        try:
                            annotate = self.annotate_bboxes_model.get_value()
                        except Exception:
                            annotate = False
                        if annotate:
                            try:
                                annotations = self._gather_annotations(self.step)
                                self.writer.write_annotations(annotations, step=self.step)
                            except Exception:
                                pass
                self.step += 1
                self.recording_time += step_size
                if self.step % 15 == 0:
                    self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"

                # handle recording-for-N-frames feature (used by 'Record 30 frames & Play')
                try:
                    cnt = getattr(self, '_record_30_remaining', None)
                    if cnt is not None:
                        if cnt > 0:
                            self._record_30_remaining = cnt - 1
                            if self._record_30_remaining <= 0:
                                try:
                                    self.disable_recording()
                                except Exception:
                                    pass
                                try:
                                    self._open_pc_player()
                                except Exception:
                                    pass
                                self._record_30_remaining = None
                except Exception:
                    pass

    def _gather_annotations(self, step: int) -> dict:
        """Collect 3D and 2D bounding boxes for prims in the stage.

        3D boxes are returned as 8-corner lists in world coordinates. 2D
        boxes are projected into the first camera found on the stage and
        reported as [xmin, ymin, xmax, ymax] in pixel coordinates. Classes
        are defaulted to the prim name if no explicit metadata is present.
        This is a best-effort helper intended for synthetic data generation.
        """
        annotations = {"bboxes2d": [], "bboxes3d": []}
        stage = get_stage()
        if stage is None:
            return annotations

        # Build bbox cache and xform cache
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Find a camera to project to (first Camera prim)
        camera_prim = None
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Camera":
                camera_prim = prim
                break

        # Determine image size and camera intrinsics if camera found
        image_w, image_h = 640, 480
        cam_params = None
        if camera_prim is not None:
            cam = UsdGeom.Camera(camera_prim)
            try:
                focal = cam.GetFocalLengthAttr().Get()
                h_ap = cam.GetHorizontalApertureAttr().Get()
                v_ap = cam.GetVerticalApertureAttr().Get()
                # assume default image size; derive focal length in pixels
                fx = focal * image_w / h_ap if (h_ap and image_w) else 1.0
                fy = focal * image_h / v_ap if (v_ap and image_h) else fx
                cx = image_w / 2.0
                cy = image_h / 2.0
                cam_params = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            except Exception:
                cam_params = None

        # Iterate prims and compute bounds
        for prim in stage.Traverse():
            if prim.IsPseudoRoot() or not prim.IsActive():
                continue
            try:
                bound = bbox_cache.ComputeWorldBound(prim)
                if bound.IsEmpty():
                    continue
                rng = bound.GetRange()
                min_pt = rng.GetMin()
                max_pt = rng.GetMax()
                # 8 corners
                corners = []
                for x in [min_pt[0], max_pt[0]]:
                    for y in [min_pt[1], max_pt[1]]:
                        for z in [min_pt[2], max_pt[2]]:
                            corners.append([float(x), float(y), float(z)])

                prim_name = prim.GetName()
                class_name = prim_name
                # 3D annotation entry
                annotations["bboxes3d"].append({
                    "prim_path": prim.GetPath().pathString,
                    "class": class_name,
                    "corners": corners,
                })

                # 2D projection (first camera only)
                if cam_params is not None and camera_prim is not None:
                    # compute world-to-camera matrix
                    cam_world = xform_cache.GetLocalToWorldTransform(camera_prim)
                    try:
                        cam_mat = cam_world.GetInverse()
                    except Exception:
                        cam_mat = None
                    if cam_mat is not None:
                        # project corners
                        xs = []
                        ys = []
                        for c in corners:
                            # Gf.Matrix4d multiply with Gf.Vec4d
                            wc = Gf.Vec4d(c[0], c[1], c[2], 1.0)
                            cc = cam_mat * wc
                            # camera coordinates: cc[0], cc[1], cc[2]
                            if cc[2] <= 0.0:
                                # behind camera: skip point
                                continue
                            x_pix = (cam_params["fx"] * (cc[0] / cc[2])) + cam_params["cx"]
                            y_pix = (cam_params["fy"] * (cc[1] / cc[2])) + cam_params["cy"]
                            xs.append(float(x_pix))
                            ys.append(float(y_pix))
                        if len(xs) > 0 and len(ys) > 0:
                            xmin = max(0.0, min(xs))
                            ymin = max(0.0, min(ys))
                            xmax = min(image_w - 1.0, max(xs))
                            ymax = min(image_h - 1.0, max(ys))
                            annotations["bboxes2d"].append({
                                "prim_path": prim.GetPath().pathString,
                                "class": class_name,
                                "bbox": [xmin, ymin, xmax, ymax],
                            })
            except Exception:
                continue

        return annotations

    def build_scenario(self):
        """Build the scenario async from UI-selected Config.

        This launches an async task which constructs the scenario (loads the
        USD stage, spawns the robot and sensors), resets the world, registers
        the physics callback and optionally starts recording.
        """

        async def _build_scenario_async():
            
            self.clear_recording()
            self.clear_scenario()

            config = self.create_config()

            self.config = config
            self.scenario = await build_scenario_from_config(config)

            self.draw_occ_map()
            
            world = get_world()
            await world.reset_async()

            self.scenario.reset()

            world.add_physics_callback("scenario_physics", self.on_physics)
            # sync UI with the newly created robot/sensors
            try:
                self._sync_ui_with_robot()
            except Exception:
                pass

            # cache stage
            self.cached_stage_path = os.path.join(tempfile.mkdtemp(), "stage.usd")
            save_stage(self.cached_stage_path)

            if self.recording_enabled:
                self.start_new_recording()

            # self.scenario.save(path)

        asyncio.ensure_future(_build_scenario_async())

    def _open_pc_player(self):
        # open a simple player window for pointcloud replay
        try:
            from omni.ext.mobility_gen.pointcloud_player import PointCloudPlayer
            # Ask user for recording path? Use last recording dir if present
            if self.writer is None:
                # pick most recent recording
                import glob, os
                recs = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*")))
                if len(recs) == 0:
                    return
                path = recs[-1]
            else:
                path = self.writer.path
            # instantiate player and store on self
            player = PointCloudPlayer(path)
            self._pc_player = player

            # Build a small UI window for player controls
            self._pc_player_window = ui.Window("PointCloud Player", width=400, height=300)
            with self._pc_player_window.frame:
                with ui.VStack():
                    with ui.HStack():
                        ui.Button("Play", clicked_fn=lambda: player.play(fps=float(self._pc_fps_model.get_value())))
                        ui.Button("Pause", clicked_fn=lambda: player.pause())
                        ui.Button("Prev", clicked_fn=lambda: player.step(forward=False))
                        ui.Button("Next", clicked_fn=lambda: player.step(forward=True))
                    with ui.HStack():
                        ui.Label("Frame")
                        self._pc_index_model = ui.SimpleIntModel(0)
                        ui.IntField(model=self._pc_index_model, height=25)
                        ui.Button("Go", clicked_fn=lambda: player.goto(int(self._pc_index_model.get_value())))
                    with ui.HStack():
                        ui.Label("FPS")
                        self._pc_fps_model = ui.SimpleIntModel(10)
                        ui.IntField(model=self._pc_fps_model, height=25, enabled=True)
                        ui.Button("Set FPS", clicked_fn=lambda: player.set_fps(float(self._pc_fps_model.get_value())))
                    ui.Label("Sensors (toggle visibility)")
                    # per-sensor toggles
                    for s in player.sensors:
                        m = ui.SimpleBoolModel(True)
                        # bind change to player visibility
                        def make_toggle(sensor_name, model):
                            def _on_change():
                                player.set_visibility(sensor_name, model.get_value())
                            return _on_change
                        cb = ui.CheckBox(model=m)
                        cb.model.add_value_changed_fn(make_toggle(s, m))
                        ui.Label(s)
            # start paused by default
            # player.play(fps=10.0)
        except Exception as e:
            print("Failed to start pointcloud player:", e)

    def _save_stage_now(self):
        try:
            if self.cached_stage_path is None:
                p = os.path.join(tempfile.mkdtemp(), "stage.usd")
            else:
                p = self.cached_stage_path
            save_stage(p)
            print(f"Stage saved to: {p}")
        except Exception as e:
            print("Failed to save stage:", e)

    def _toggle_occ_map(self):
        try:
            # try both APIs: set_visible or .visible attribute
            if hasattr(self._visualize_window, 'visible'):
                try:
                    self._visualize_window.visible = not self._visualize_window.visible
                    return
                except Exception:
                    pass
            if hasattr(self._visualize_window, 'set_visible'):
                try:
                    self._visualize_window.set_visible(not self._visualize_window.get_visible())
                    return
                except Exception:
                    pass
        except Exception:
            pass

    def _show_help(self):
        try:
            print("MobilityGen extension help:\n - Build Scenario: build the chosen scene and robot.\n - Plan & Start Auto: plan a path and start autonomous following using the scenario if available.\n - Record PointClouds: enable automatic pointcloud capture during physics stepping.\n")
        except Exception:
            pass

    def _scan_worlds(self):
        """Search common locations for USD/world files to populate the Worlds combo.

        Returns a list of absolute paths (may be empty).
        """
        exts = ('.usd', '.usda', '.usdc')
        roots = [os.getcwd()]
        # allow user-provided worlds dir (UI) to be used for scanning
        try:
            try:
                # prefer the UI-provided value if present
                v = None
                if hasattr(self, 'worlds_dir_field') and self.worlds_dir_field is not None:
                    try:
                        v = self.worlds_dir_field.model.get_value()
                    except Exception:
                        try:
                            v = self.worlds_dir_field.model.as_string
                        except Exception:
                            v = None
                if v:
                    v = os.path.expanduser(v)
                    if os.path.isdir(v):
                        roots.append(v)
            except Exception:
                pass
            if DATA_DIR and os.path.isdir(DATA_DIR):
                roots.append(DATA_DIR)
        except Exception:
            pass

        found = []
        for r in roots:
            try:
                for root, dirs, files in os.walk(r):
                    for f in files:
                        if f.lower().endswith(exts):
                            found.append(os.path.join(root, f))
            except Exception:
                continue

        # dedupe and sort
        uniq = []
        seen = set()
        for p in found:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return sorted(uniq)[:64]

    def _refresh_worlds(self):
        try:
            self._available_worlds = self._scan_worlds()
            items = self._available_worlds if len(self._available_worlds) > 0 else ["(none found)"]
            # rebuild combo model if possible
            try:
                # attempt to reset items by creating a new ComboBox model value
                self.worlds_combo_box = ui.ComboBox(0, *items)
            except Exception:
                pass
            print(f"Refreshed worlds, found {len(self._available_worlds)} files")
        except Exception as e:
            print("Failed to refresh worlds:", e)

    def _apply_selected_world(self):
        try:
            if not hasattr(self, 'worlds_combo_box'):
                return
            idx = self.worlds_combo_box.model.get_item_value_model().get_value_as_int()
            items = self._available_worlds if len(self._available_worlds) > 0 else ["(none found)"]
            if idx < 0 or idx >= len(items):
                return
            sel = items[idx]
            if sel and sel != "(none found)":
                self.scene_usd_field.model.set_value(sel)
        except Exception as e:
            print("Failed to apply selected world:", e)

    # --- sensor quick helpers ---
    def _enable_all_cameras(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            for name in ('front_stereo', 'fisheye_left', 'fisheye_right'):
                try:
                    mod = getattr(robot, name, None)
                    if mod is None:
                        continue
                    if hasattr(mod, 'enable_rgb_rendering'):
                        mod.enable_rgb_rendering()
                    else:
                        if hasattr(mod, 'left') and hasattr(mod.left, 'enable_rgb_rendering'):
                            mod.left.enable_rgb_rendering()
                        if hasattr(mod, 'right') and hasattr(mod.right, 'enable_rgb_rendering'):
                            mod.right.enable_rgb_rendering()
                except Exception:
                    pass
            print('Enabled all cameras (best-effort)')
        except Exception as e:
            print('Error enabling cameras:', e)

    def _disable_all_cameras(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            for name in ('front_stereo', 'fisheye_left', 'fisheye_right'):
                try:
                    mod = getattr(robot, name, None)
                    if mod is None:
                        continue
                    if hasattr(mod, 'disable_rendering'):
                        mod.disable_rendering()
                    else:
                        if hasattr(mod, 'left') and hasattr(mod.left, 'disable_rendering'):
                            mod.left.disable_rendering()
                        if hasattr(mod, 'right') and hasattr(mod.right, 'disable_rendering'):
                            mod.right.disable_rendering()
                except Exception:
                    pass
            print('Disabled all cameras (best-effort)')
        except Exception as e:
            print('Error disabling cameras:', e)

    def _enable_all_lidar(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            try:
                mod = getattr(robot, 'lidar', None)
                if mod is not None and hasattr(mod, 'enable_lidar'):
                    mod.enable_lidar()
            except Exception:
                pass
            print('Enabled lidar (best-effort)')
        except Exception as e:
            print('Error enabling lidar:', e)

    def _disable_all_lidar(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            try:
                mod = getattr(robot, 'lidar', None)
                if mod is not None and hasattr(mod, 'disable_lidar'):
                    mod.disable_lidar()
            except Exception:
                pass
            print('Disabled lidar (best-effort)')
        except Exception as e:
            print('Error disabling lidar:', e)

    # ------------------ Robot / sensor UI handlers ------------------
    def _apply_control_mode(self):
        """Apply the control mode chosen in the UI to the active robot.

        This attempts to call a `set_control_mode` method on the robot and
        falls back to setting a `control_mode` attribute if present.
        """
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            idx = self.control_mode_combo.model.get_item_value_model().get_value_as_int()
            mode = "manual" if idx == 0 else "auto"
            try:
                robot.set_control_mode(mode)
            except Exception:
                # fallback: try setting attribute
                try:
                    robot.control_mode = mode
                except Exception:
                    pass
        except Exception:
            pass

    def _plan_path_and_set_auto(self):
        """Plan a path using the robot or scenario and enable autonomous mode.

        Behavior:
          - If the active scenario exposes `target_path` (a Buffer-like), we
            prefer to set that so the scenario's built-in follower handles
            control. We also set a PathHelper for the scenario.
          - Otherwise, ask the robot to plan a path and set the robot into
            'auto' control mode.
        """
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            occupancy_map = getattr(self.scenario, 'occupancy_map', None)
            if occupancy_map is None:
                return

            # Prefer driving the scenario (if it exposes a target_path Buffer)
            # so that the scenario's path-following logic is reused. Otherwise
            # fall back to the robot convenience planner.
            try:
                # If the scenario has a target_path Buffer, set it and also
                # update its PathHelper so the follower can use it.
                if hasattr(self.scenario, 'target_path'):
                    try:
                        path_list = robot.plan_path_from_occupancy_map(occupancy_map)
                    except Exception:
                        # If robot planner fails, try letting the scenario generate
                        # its own random target if it exposes that method.
                        if hasattr(self.scenario, '_set_random_target_path'):
                            try:
                                self.scenario._set_random_target_path()
                                # ensure UI reflects auto mode
                                try:
                                    robot.set_control_mode('auto')
                                except Exception:
                                    try:
                                        robot.control_mode = 'auto'
                                    except Exception:
                                        pass
                                return
                            except Exception as e:
                                raise e
                        raise

                    import numpy as _np
                    arr = _np.asarray(path_list, dtype=_np.float32)
                    try:
                        # Buffer-like API
                        self.scenario.target_path.set_value(arr)
                    except Exception:
                        try:
                            self.scenario.target_path = arr
                        except Exception:
                            pass
                    try:
                        # update helper used by the scenario
                        self.scenario._helper = PathHelper(arr)
                    except Exception:
                        pass
                    try:
                        robot.set_control_mode('auto')
                    except Exception:
                        try:
                            robot.control_mode = 'auto'
                        except Exception:
                            pass
                    return

                # fallback: ask robot to plan and set its own auto-path
                path = robot.plan_path_from_occupancy_map(occupancy_map)
                try:
                    robot.set_control_mode('auto')
                except Exception:
                    try:
                        robot.control_mode = 'auto'
                    except Exception:
                        pass
            except Exception as e:
                print('Path planning failed:', e)
        except Exception:
            pass

    def _toggle_sensor(self, sensor_name: str, enabled: bool):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            module = getattr(robot, sensor_name, None)
            if module is None:
                return
            # cameras: Camera instances expose enable_rgb_rendering()/disable_rendering()
            if enabled:
                try:
                    if hasattr(module, 'enable_rgb_rendering'):
                        module.enable_rgb_rendering()
                    else:
                        # stereo wrapper may have left/right
                        if hasattr(module, 'left') and hasattr(module.left, 'enable_rgb_rendering'):
                            module.left.enable_rgb_rendering()
                        if hasattr(module, 'right') and hasattr(module.right, 'enable_rgb_rendering'):
                            module.right.enable_rgb_rendering()
                except Exception:
                    pass
                try:
                    if hasattr(module, 'enable_lidar'):
                        module.enable_lidar()
                except Exception:
                    pass
            else:
                try:
                    if hasattr(module, 'disable_rendering'):
                        module.disable_rendering()
                    else:
                        if hasattr(module, 'left') and hasattr(module.left, 'disable_rendering'):
                            module.left.disable_rendering()
                        if hasattr(module, 'right') and hasattr(module.right, 'disable_rendering'):
                            module.right.disable_rendering()
                except Exception:
                    pass
                try:
                    if hasattr(module, 'disable_lidar'):
                        module.disable_lidar()
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_ui_with_robot(self):
        """Read scenario.robot and update UI models to reflect current sensor/control state."""
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return

            # control mode
            try:
                mode = getattr(robot, 'control_mode', None)
                if mode is None and hasattr(robot, 'control_mode'):
                    mode = robot.control_mode
                if mode is not None:
                    idx = 0 if mode == 'manual' else 1
                    try:
                        self.control_mode_combo.model.set_value(idx)
                    except Exception:
                        pass
            except Exception:
                pass

            # sensors: set checkbox models if sensors exist
            try:
                self._sensor_front_model.set_value(bool(getattr(robot, 'front_stereo', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_left_model.set_value(bool(getattr(robot, 'fisheye_left', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_right_model.set_value(bool(getattr(robot, 'fisheye_right', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_lidar_model.set_value(bool(getattr(robot, 'lidar', None) is not None))
            except Exception:
                pass
        except Exception:
            pass

    def _record_30_and_play(self):
        """Start recording for 30 frames and then open the PC player automatically."""
        try:
            if self.scenario is None:
                return
            # ensure writer exists and recording enabled
            if not self.recording_enabled:
                self.start_new_recording()
                self.recording_enabled = True
            # set countdown
            self._record_30_remaining = 30
        except Exception as e:
            print('Failed to start 30-frame recording:', e)