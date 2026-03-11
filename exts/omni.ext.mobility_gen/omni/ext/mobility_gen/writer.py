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
import json

try:
    import open3d as o3d
except Exception:
    o3d = None

from omni.ext.mobility_gen.config import Config
from omni.ext.mobility_gen.occupancy_map import OccupancyMap

# Enable verbose writer debugging via environment variable MOBILITY_GEN_DEBUG
# default to '1' to make debug prints visible during diagnostics; set to '0'
# to silence these messages in normal runs.
DEBUG_WRITER = os.environ.get("MOBILITY_GEN_DEBUG", "1") == "1"


class Writer:

    def __init__(self, path: str):
        self.path = path
        # Ensure common annotation folders exist so downstream tools/readers
        # find them even before any annotations are written.
        try:
            for d in ("bboxes2d", "bboxes3d", "classes"):
                os.makedirs(os.path.join(self.path, "state", d), exist_ok=True)
        except Exception:
            # Best-effort: ignore filesystem errors here
            pass
        if DEBUG_WRITER:
            print(f"[Writer] Initialized writer at {self.path}")

    def write_state_dict_common(self, state_dict: dict, step: int):
        dict_folder = os.path.join(self.path, "state", "common")
        if not os.path.exists(dict_folder):
            os.makedirs(dict_folder)
        state_dict_path = os.path.join(dict_folder, f"{step:08d}.npy")
        try:
            np.save(state_dict_path, state_dict)
            if DEBUG_WRITER:
                print(f"[Writer] Saved common state step={step} -> {state_dict_path}")
        except Exception as e:
            print(f"[Writer] Error saving common state step={step} -> {state_dict_path}: {e}")

    def write_state_dict_rgb(self, state_rgb: dict, step: int):
        for name, value in state_rgb.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "rgb", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.jpg")
                image = PIL.Image.fromarray(value)
                image.save(image_path)
                if DEBUG_WRITER:
                    print(f"[Writer] Saved RGB '{name}' step={step} -> {image_path}")

    @staticmethod
    def _save_segmentation_png(arr: np.ndarray, image_path: str):
        """Save a segmentation array as a PNG, handling all dtype/shape edge cases.

        The PIL version bundled with Isaac Sim has bugs with modes 'F', 'I',
        and 'I;16' when saving PNG.  We always produce a uint8 array and
        ensure the buffer is C-contiguous before calling PIL.

        If PIL still fails, we fall back to saving as .npy.
        """
        arr = np.asarray(arr)

        # Squeeze ALL singleton dimensions (e.g. (H, W, 1, 1) -> (H, W))
        arr = arr.squeeze()

        # Ensure basic numeric dtype
        if arr.dtype.kind not in ('u', 'i', 'f', 'b'):
            arr = arr.astype(np.int64)

        # Float -> int
        if arr.dtype.kind == 'f':
            arr = np.rint(arr).astype(np.int64)

        # Force to a standard signed/unsigned int if not already
        if arr.dtype.kind == 'i':
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == 'u' and arr.dtype.itemsize > 1:
            arr = arr.astype(np.uint32)

        # Convert to uint8-safe representation
        if arr.ndim == 2:
            mn = int(arr.min()) if arr.size > 0 else 0
            mx = int(arr.max()) if arr.size > 0 else 0
            # Shift negative values to 0-based range
            if mn < 0:
                arr = arr - mn
                mx = mx - mn
            if mx <= 255:
                arr = arr.astype(np.uint8)
            else:
                # Encode as 3-channel uint8 (R=low, G=mid, B=high byte)
                flat = arr.astype(np.uint32)
                r = (flat & 0xFF).astype(np.uint8)
                g = ((flat >> 8) & 0xFF).astype(np.uint8)
                b = ((flat >> 16) & 0xFF).astype(np.uint8)
                arr = np.stack([r, g, b], axis=-1)
        elif arr.ndim >= 3:
            # Multi-channel – cast each channel to uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.ndim < 2:
            # Scalar or 1D – cannot save as image; go straight to .npy
            npy_path = image_path.rsplit('.', 1)[0] + '.npy'
            np.save(npy_path, arr)
            print(f'[Writer] WARNING: segmentation data is {arr.ndim}D; saved as {npy_path}')
            return

        # Guarantee C-contiguous memory layout (fixes PIL stride bugs)
        arr = np.ascontiguousarray(arr)

        try:
            image = PIL.Image.fromarray(arr)
            image.save(image_path)
        except Exception as e:
            # Last-resort fallback: save as .npy so data is not lost
            npy_path = image_path.rsplit('.', 1)[0] + '.npy'
            np.save(npy_path, arr)
            print(f'[Writer] WARNING: PIL PNG save failed ({e}); saved as {npy_path}')

    def write_state_dict_segmentation(self, state_segmentation: dict, step: int):
        for name, value in state_segmentation.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "segmentation", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.png")
                self._save_segmentation_png(value, image_path)
                if DEBUG_WRITER:
                    print(f"[Writer] Saved segmentation '{name}' step={step} -> {image_path}")

    def write_state_dict_instance_id_segmentation(self, state_segmentation: dict, step: int):
        for name, value in state_segmentation.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "instance_id_segmentation", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.png")
                self._save_segmentation_png(value, image_path)
                if DEBUG_WRITER:
                    print(f"[Writer] Saved instance-id segmentation '{name}' step={step} -> {image_path}")

    def write_state_dict_depth(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "depth", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Inverse depth 16bit
                inverse_depth = 1.0 / (1.0 + value)
                inverse_depth = (65535 * inverse_depth).astype(np.uint16)
                inverse_depth = np.ascontiguousarray(inverse_depth)

                output_path_png = os.path.join(output_folder, f"{step:08d}.png")
                output_path_npy = os.path.join(output_folder, f"{step:08d}.npy")

                try:
                    image = PIL.Image.fromarray(inverse_depth, "I;16")
                    image.save(output_path_png)
                except Exception:
                    # PIL cannot save I;16 as PNG in this version; save as .npy
                    np.save(output_path_npy, inverse_depth)

                if DEBUG_WRITER:
                    print(f"[Writer] Saved depth '{name}' step={step}")

    def write_state_dict_normals(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "normals", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, f"{step:08d}.npy")
                try:
                    np.save(output_path, value)
                    if DEBUG_WRITER:
                        print(f"[Writer] Saved normals '{name}' step={step} -> {output_path}")
                except Exception as e:
                    print(f"[Writer] Error saving normals '{name}' step={step} -> {output_path}: {e}")

    def _save_ply(self, points: np.ndarray, path: str):
        """Write a simple ASCII PLY file with XYZ (and optional intensity).

        This is a minimal writer to avoid adding a heavy dependency. It writes
        an ASCII PLY with vertex properties x y z [intensity].
        """
        if points is None:
            return
        # Ensure Nx3 or Nx4
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be Nx3 or Nx4")

        # If open3d is available, prefer to write a binary PLY/PCD for efficiency
        if o3d is not None:
            try:
                # build point cloud
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
                if pts.shape[1] >= 6:
                    # rgb in cols 3-5
                    rgb = pts[:, 3:6]
                    rgb = np.asarray(rgb)
                    # normalize to 0..1 if necessary
                    if rgb.max() > 1.0:
                        rgb = rgb / 255.0
                    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
                elif pts.shape[1] == 4:
                    # intensity only: map to grayscale color
                    intensity = pts[:, 3]
                    intensity = np.asarray(intensity).astype(np.float64)
                    if intensity.max() > 1.0:
                        intensity = intensity / (intensity.max() + 1e-6)
                    gray = np.stack([intensity, intensity, intensity], axis=1)
                    pc.colors = o3d.utility.Vector3dVector(gray)

                # write binary PLY by default
                o3d.io.write_point_cloud(path, pc, write_ascii=False)
                return
            except Exception:
                # fallback to ASCII writer below
                pass

        # Fallback ASCII PLY writer
        with open(path, "w") as f:
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {pts.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
            ]
            # support intensity (4) or rgb (6) or rgb+intensity (7)
            if pts.shape[1] == 4:
                header.append("property float intensity")
            elif pts.shape[1] == 6:
                header.extend(["property uchar red", "property uchar green", "property uchar blue"])
            elif pts.shape[1] >= 7:
                header.extend(["property uchar red", "property uchar green", "property uchar blue", "property float intensity"])
            header.append("end_header")
            f.write("\n".join(header) + "\n")
            for p in pts:
                if pts.shape[1] == 4:
                    f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
                elif pts.shape[1] == 6:
                    # rgb may be floats [0-1] or ints [0-255]
                    r, g, b = p[3], p[4], p[5]
                    if float(r) <= 1.0 and float(g) <= 1.0 and float(b) <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(r)} {int(g)} {int(b)}\n")
                elif pts.shape[1] >= 7:
                    r, g, b = p[3], p[4], p[5]
                    if float(r) <= 1.0 and float(g) <= 1.0 and float(b) <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(r)} {int(g)} {int(b)} {p[6]}\n")
                else:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")

    def write_annotations(self, annotations: dict, step: int):
        """Write annotations into split folders under state/.

        Output layout:
          - state/bboxes2d/<step>.json
          - state/bboxes3d/<step>.json
          - state/classes/<step>.json
          - state/semantic/<step>.json
        """
        payload = annotations if isinstance(annotations, dict) else {}
        self._write_json_state_entry("bboxes2d", payload.get("bboxes2d", []), step)
        self._write_json_state_entry("bboxes3d", payload.get("bboxes3d", []), step)
        self._write_json_state_entry("classes", payload.get("classes", []), step)
        self._write_json_state_entry("semantic", payload.get("semantic", {}), step)

    def _write_json_state_entry(self, folder_name: str, data, step: int):
        folder = os.path.join(self.path, "state", folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_path = os.path.join(folder, f"{step:08d}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def write_pointcloud_metadata(self, metadata: dict, step: int):
        """Write per-sensor metadata for pointclouds.

        metadata is expected to be a dict mapping sensor full names to a small
        dict with keys like 'position' and 'orientation'. We write each
        sensor's metadata next to the point cloud files as
        state/point_cloud/<sensor_name>/<step>_meta.json
        """
        for sensor_name, meta in metadata.items():
            output_folder = os.path.join(self.path, "state", "point_cloud", sensor_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            meta_path = os.path.join(output_folder, f"{step:08d}_meta.json")
            try:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                if DEBUG_WRITER:
                    print(f"[Writer] Saved pointcloud metadata for {sensor_name} step={step} -> {meta_path}")
            except Exception:
                # Best-effort: ignore failures
                pass

    def write_state_dict_pointcloud(self, state_pc: dict, step: int, save_format: str = "npy"):
        """Write point-cloud data to disk.

        Each point-cloud value is expected to be a Nx3 or Nx4 numpy array.
        By default we save as .npy; set save_format to 'ply' to write ASCII
        PLY files alongside the .npy (or instead).
        """
        if DEBUG_WRITER:
            try:
                sensors = list(state_pc.keys()) if isinstance(state_pc, dict) else str(type(state_pc))
            except Exception:
                sensors = 'unknown'
            print(f"[Writer] write_state_dict_pointcloud step={step} sensors={sensors} save_format={save_format}")

        for name, value in state_pc.items():
            if value is None:
                if DEBUG_WRITER:
                    print(f"[Writer] Pointcloud '{name}' is None at step {step}, skipping")
                continue

            output_folder = os.path.join(self.path, "state", "point_cloud", name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # try to determine number of points for logging
            try:
                pts = np.asarray(value)
                npts = pts.shape[0] if pts.ndim == 2 else -1
            except Exception:
                pts = None
                npts = 'unknown'

            if DEBUG_WRITER:
                print(f"[Writer] Writing pointcloud '{name}' step={step} npts={npts} into {output_folder} (format={save_format})")

            try:
                if save_format == "npy":
                    output_path = os.path.join(output_folder, f"{step:08d}.npy")
                    np.save(output_path, value)
                    if DEBUG_WRITER:
                        print(f"[Writer] Saved pointcloud (npy): {output_path}")
                elif save_format == "ply":
                    output_path = os.path.join(output_folder, f"{step:08d}.ply")
                    try:
                        self._save_ply(value, output_path)
                        if DEBUG_WRITER:
                            print(f"[Writer] Saved pointcloud (ply): {output_path}")
                    except Exception as e:
                        fallback_npy = os.path.join(output_folder, f"{step:08d}.npy")
                        np.save(fallback_npy, value)
                        print(f"[Writer] _save_ply failed for {name} step {step}: {e}, saved npy fallback {fallback_npy}")
                elif save_format == "pcd":
                    output_path = os.path.join(output_folder, f"{step:08d}.pcd")
                    try:
                        if o3d is not None:
                            pc = o3d.geometry.PointCloud()
                            pc.points = o3d.utility.Vector3dVector(np.asarray(value)[:, :3].astype(np.float64))
                            if np.asarray(value).shape[1] >= 6:
                                rgb = np.asarray(value)[:, 3:6]
                                if rgb.max() > 1.0:
                                    rgb = rgb / 255.0
                                pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
                            elif np.asarray(value).shape[1] == 4:
                                intensity = np.asarray(value)[:, 3]
                                if intensity.max() > 1.0:
                                    intensity = intensity / (intensity.max() + 1e-6)
                                gray = np.stack([intensity, intensity, intensity], axis=1)
                                pc.colors = o3d.utility.Vector3dVector(gray)
                            o3d.io.write_point_cloud(output_path, pc, write_ascii=False)
                            if DEBUG_WRITER:
                                print(f"[Writer] Saved pointcloud (pcd): {output_path}")
                        else:
                            ply_path = os.path.join(output_folder, f"{step:08d}.ply")
                            self._save_ply(value, ply_path)
                            if DEBUG_WRITER:
                                print(f"[Writer] Saved pointcloud (ply fallback): {ply_path}")
                    except Exception as e:
                        fallback_npy = os.path.join(output_folder, f"{step:08d}.npy")
                        np.save(fallback_npy, value)
                        print(f"[Writer] PCD write failed for {name} step {step}: {e}, saved npy fallback {fallback_npy}")
                else:
                    # unknown format: default to npy
                    output_path = os.path.join(output_folder, f"{step:08d}.npy")
                    np.save(output_path, value)
                    if DEBUG_WRITER:
                        print(f"[Writer] Saved pointcloud (default npy): {output_path}")
            except Exception as e:
                print(f"[Writer] Error writing pointcloud '{name}' step {step}: {e}")

    # Alias for callers using underscore-separated name
    write_state_dict_point_cloud = write_state_dict_pointcloud

    # Backwards-compatible alias: some callers use the name "lidar"
    write_state_dict_lidar = write_state_dict_pointcloud

    def write_stage(self):
        from omni.ext.mobility_gen.utils.global_utils import save_stage

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

    def copy_init(self, other_path: str, overwrite: bool = False, verbose: bool = False):
        """Copy initial artifacts (stage/config/occupancy_map) from another recording.

        Args:
            other_path: source recording directory
            overwrite: if True, remove destination path before copying
            verbose: if True, print progress messages
        """
        if overwrite and os.path.exists(self.path):
            try:
                if verbose:
                    print(f"[Writer] Overwrite requested: removing existing output path {self.path}")
                shutil.rmtree(self.path)
            except Exception:
                # best-effort: continue
                pass

        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass

        # Copy stage and config if they exist in the source recording. Be robust
        # when files or folders are missing or destination already exists.
        try:
            src_stage = os.path.join(other_path, "stage.usd")
            dst_stage = os.path.join(self.path, "stage.usd")
            if os.path.exists(src_stage):
                shutil.copyfile(src_stage, dst_stage)
                if verbose:
                    print(f"[Writer] Copied stage.usd to {dst_stage}")
        except Exception:
            # best-effort
            if verbose:
                print(f"[Writer] Failed to copy stage.usd from {other_path}")

        try:
            src_cfg = os.path.join(other_path, "config.json")
            dst_cfg = os.path.join(self.path, "config.json")
            if os.path.exists(src_cfg):
                shutil.copyfile(src_cfg, dst_cfg)
                if verbose:
                    print(f"[Writer] Copied config.json to {dst_cfg}")
        except Exception:
            if verbose:
                print(f"[Writer] Failed to copy config.json from {other_path}")

        # occupancy_map may be large and the destination could already exist
        try:
            src_occ = os.path.join(other_path, "occupancy_map")
            dst_occ = os.path.join(self.path, "occupancy_map")
            if os.path.exists(src_occ):
                if os.path.exists(dst_occ):
                    if verbose:
                        print(f"[Writer] Merging occupancy_map from {src_occ} into existing {dst_occ}")
                    # merge files from src into dst (overwrite existing files)
                    for root, dirs, files in os.walk(src_occ):
                        rel = os.path.relpath(root, src_occ)
                        dest_dir = os.path.join(dst_occ, rel) if rel != '.' else dst_occ
                        os.makedirs(dest_dir, exist_ok=True)
                        for f in files:
                            try:
                                shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                if verbose:
                                    print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                            except Exception:
                                # ignore individual file copy failures
                                if verbose:
                                    print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                pass
                else:
                    # typical case: simply copy the whole tree
                    try:
                        shutil.copytree(src_occ, dst_occ)
                        if verbose:
                            print(f"[Writer] Copied occupancy_map tree to {dst_occ}")
                    except Exception:
                        # fallback to manual copy
                        if verbose:
                            print(f"[Writer] copytree failed, falling back to manual copy for occupancy_map")
                        for root, dirs, files in os.walk(src_occ):
                            rel = os.path.relpath(root, src_occ)
                            dest_dir = os.path.join(dst_occ, rel) if rel != '.' else dst_occ
                            os.makedirs(dest_dir, exist_ok=True)
                            for f in files:
                                try:
                                    shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                    if verbose:
                                        print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                                except Exception:
                                    if verbose:
                                        print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                    pass
        except Exception:
            if verbose:
                print(f"[Writer] Error while copying occupancy_map from {other_path}")

        # Copy annotation folders (bboxes2d, bboxes3d, classes) if present
        try:
            for ann in ("bboxes2d", "bboxes3d", "classes"):
                src_ann = os.path.join(other_path, "state", ann)
                dst_ann = os.path.join(self.path, "state", ann)
                if os.path.exists(src_ann):
                    if os.path.exists(dst_ann):
                        if verbose:
                            print(f"[Writer] Merging {src_ann} into existing {dst_ann}")
                        for root, dirs, files in os.walk(src_ann):
                            rel = os.path.relpath(root, src_ann)
                            dest_dir = os.path.join(dst_ann, rel) if rel != '.' else dst_ann
                            os.makedirs(dest_dir, exist_ok=True)
                            for f in files:
                                try:
                                    shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                    if verbose:
                                        print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                                except Exception:
                                    if verbose:
                                        print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                    pass
                    else:
                        try:
                            shutil.copytree(src_ann, dst_ann)
                            if verbose:
                                print(f"[Writer] Copied annotation tree {src_ann} -> {dst_ann}")
                        except Exception:
                            if verbose:
                                print(f"[Writer] copytree failed for {src_ann}, falling back to manual copy")
                            for root, dirs, files in os.walk(src_ann):
                                rel = os.path.relpath(root, src_ann)
                                dest_dir = os.path.join(dst_ann, rel) if rel != '.' else dst_ann
                                os.makedirs(dest_dir, exist_ok=True)
                                for f in files:
                                    try:
                                        shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                        if verbose:
                                            print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                                    except Exception:
                                        if verbose:
                                            print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                        pass
        except Exception:
            if verbose:
                print(f"[Writer] Error while copying annotation folders from {other_path}")