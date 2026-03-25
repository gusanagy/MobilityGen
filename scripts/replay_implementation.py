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

"""

This script launches a simulation app for replaying and rendering
a recording.

"""

from isaacsim import SimulationApp

# Enable MotionBVH so lidar models' motion-corrected BVH is used by RTX lidar.
simulation_app = SimulationApp(
    launch_config={"headless": True, "enable_motion_bvh": True}
)

import argparse
import glob
import os
import shutil

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import tqdm
from omni.ext.mobility_gen.build import load_scenario
from omni.ext.mobility_gen.reader import Reader
from omni.ext.mobility_gen.utils.global_utils import get_world
from omni.ext.mobility_gen.writer import Writer
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--rgb_enabled", type=bool, default=True)
    parser.add_argument("--segmentation_enabled", type=bool, default=True)
    parser.add_argument("--depth_enabled", type=bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=bool, default=True)
    parser.add_argument("--normals_enabled", type=bool, default=False)
    parser.add_argument("--lidar_enabled", type=bool, default=True)
    parser.add_argument("--render_rt_subframes", type=int, default=5)
    parser.add_argument("--render_interval", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="Limit number of replay iterations (diagnostic)")

    args, unknown = parser.parse_known_args()

    import time as _time

    print(f"[SETUP] load_scenario ...")
    _t0 = _time.time()
    scenario = load_scenario(os.path.join(args.input_path))
    print(f"[SETUP] load_scenario done ({_time.time() - _t0:.1f}s)")

    print(f"[SETUP] get_world ...")
    _t0 = _time.time()
    world = get_world()
    print(f"[SETUP] get_world done ({_time.time() - _t0:.1f}s)")

    print(f"[SETUP] world.reset ...")
    _t0 = _time.time()
    try:
        world.reset()
        print(f"[SETUP] world.reset done ({_time.time()-_t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[ERROR] world.reset failed: {e}", flush=True)
        raise

    print(scenario, flush=True)

    if args.rgb_enabled:
        print(f"[SETUP] enable_rgb_rendering ...")
        _t0 = _time.time()
        scenario.enable_rgb_rendering()
        print(f"[SETUP] enable_rgb_rendering done ({_time.time()-_t0:.1f}s)")

    if args.segmentation_enabled:
        print(f"[SETUP] enable_segmentation_rendering ...")
        _t0 = _time.time()
        scenario.enable_segmentation_rendering()
        print(f"[SETUP] enable_segmentation_rendering done ({_time.time()-_t0:.1f}s)")

    if args.depth_enabled:
        print(f"[SETUP] enable_depth_rendering ...")
        _t0 = _time.time()
        scenario.enable_depth_rendering()
        print(f"[SETUP] enable_depth_rendering done ({_time.time() - _t0:.1f}s)")

    if args.instance_id_segmentation_enabled:
        print(f"[SETUP] enable_instance_id_segmentation_rendering ...")
        _t0 = _time.time()
        scenario.enable_instance_id_segmentation_rendering()
        print(
            f"[SETUP] enable_instance_id_segmentation_rendering done ({_time.time() - _t0:.1f}s)"
        )

    if args.normals_enabled:
        print(f"[SETUP] enable_normals_rendering ...")
        _t0 = _time.time()
        scenario.enable_normals_rendering()
        print(f"[SETUP] enable_normals_rendering done ({_time.time() - _t0:.1f}s)")

    if args.lidar_enabled:
        print(f"[SETUP] enable_lidar_rendering ...")
        _t0 = _time.time()
        scenario.enable_lidar_rendering()
        print(f"[SETUP] enable_lidar_rendering done ({_time.time()-_t0:.1f}s)")

    # Warm-up: use multiple simulation_app.update() calls to let the
    # renderer compile shaders and build acceleration structures gradually,
    # instead of a single rep.orchestrator.step() which tries to render ALL
    # annotators at once and can block on a memory-constrained GPU.
    _WARMUP_FRAMES = int(os.environ.get("MOBILITY_GEN_WARMUP_FRAMES", "5"))
    print(f"[SETUP] Warming up renderer ({_WARMUP_FRAMES}x simulation_app.update) ...")
    _t0 = _time.time()
    for _wi in range(_WARMUP_FRAMES):
        simulation_app.update()
        print(f"[SETUP]   warmup frame {_wi+1}/{_WARMUP_FRAMES} ({_time.time()-_t0:.1f}s)")
    print(f"[SETUP] Warmup done ({_time.time()-_t0:.1f}s)")

    # Step the world once to ensure all physics handles (ArticulationView,
    # etc.) are fully initialised — simulation_app.update() alone doesn't
    # step the physics pipeline.
    world.step(render=False)
    print("[SETUP] world.step done – physics handles ready")

    reader = Reader(args.input_path)
    num_steps = len(reader)
    print(f"[DEBUG] Reader created, num_steps={num_steps}")

    writer = Writer(args.output_path)
    print(f"[DEBUG] Writer created at {args.output_path}")
    print(f"[DEBUG] writer.copy_init starting")
    writer.copy_init(args.input_path)
    print(f"[DEBUG] writer.copy_init done")

    print(f"============== Replaying ==============")
    print(f"\tInput path: {args.input_path}")
    print(f"\tOutput path: {args.output_path}")
    print(f"\tRgb enabled: {args.rgb_enabled}")
    print(f"\tSegmentation enabled: {args.segmentation_enabled}")
    print(f"\tRendering RT subframes: {args.render_rt_subframes}")
    print(f"\tRender interval: {args.render_interval}")

    for i, step in enumerate(tqdm.tqdm(range(0, num_steps, args.render_interval))):
        if args.max_steps is not None and i >= args.max_steps:
            break

        _loop_t0 = _time.time()

        print(f"[LOOP] step {step} (iter {i}): begin")
        print(f"[LOOP] step {step}: read_state_dict")
        state_dict = reader.read_state_dict(index=step)
        print(f"[LOOP] step {step}: load_state_dict")
        scenario.load_state_dict(state_dict)
        print(f"[LOOP] step {step}: write_replay_data")
        try:
            scenario.write_replay_data()
        except Exception as e:
            print(f"[WARNING] scenario.write_replay_data failed at step {step}: {e}")

        print(f"[LOOP] step {step}: do render frame updates")
        for _sf in range(args.render_rt_subframes + 1):
            t0 = _time.time()
            simulation_app.update()
            dt = _time.time() - t0
            if dt > 5.0:
                print(f"[WARNING] simulation_app.update() took {dt:.1f}s on render subframe {_sf} (step {step})")
        print(
            f"[LOOP] step {step} (iter {i}): render done ({_time.time() - _loop_t0:.2f}s)"
        )

        t0_state = _time.time()
        try:
            scenario.update_state()
        except Exception as e:
            print(f"[WARNING] scenario.update_state failed at step {step}: {e}")
        took_state = _time.time() - t0_state
        if took_state > 10.0:
            print(f"[WARNING] update_state took {took_state:.1f}s at step {step}")
        print(
            f"[LOOP] step {step} (iter {i}): update_state done ({_time.time() - _loop_t0:.2f}s)"
        )

        state_dict = scenario.state_dict_common()
        state_rgb = scenario.state_dict_rgb()
        state_segmentation = scenario.state_dict_segmentation()
        state_depth = scenario.state_dict_depth()
        state_normals = scenario.state_dict_normals()
        state_point_cloud = scenario.state_dict_point_cloud()

        print(f"[DEBUG] Loop {i}, step {step}: writing outputs ...")
        writer.write_state_dict_common(state_dict, step)
        writer.write_state_dict_rgb(state_rgb, step)
        writer.write_state_dict_segmentation(state_segmentation, step)
        writer.write_state_dict_depth(state_depth, step)
        writer.write_state_dict_normals(state_normals, step)
        writer.write_state_dict_point_cloud(state_point_cloud, step)

    print("[DONE] Replay finished successfully.")
    simulation_app.close()
    # Force exit: simulation_app.close() can hang during shutdown
    # (X connection broken, GPU resource cleanup deadlock, etc.)
    import os as _os

    _os._exit(0)
