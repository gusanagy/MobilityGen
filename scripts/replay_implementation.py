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

print("[DEBUG] Importing SimulationApp...", flush=True)
try:
    from isaacsim import SimulationApp
    print("[DEBUG] SimulationApp imported.", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import SimulationApp: {e}", flush=True)
    raise

# Enable MotionBVH so lidar models' motion-corrected BVH is used by RTX lidar.
print("[DEBUG] Creating SimulationApp...", flush=True)
try:
    simulation_app = SimulationApp(launch_config={"headless": True, "enable_motion_bvh": True})
    print("[DEBUG] SimulationApp created.", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to create SimulationApp: {e}", flush=True)
    raise

import argparse
import os
import shutil
import numpy as np
from PIL import Image
import glob
import tqdm

import omni.replicator.core as rep

# Attempt normal import; if running directly (without --ext-folder/--enable)
# the omni.ext.mobility_gen package may not be on sys.path. Try to
# auto-inject the local `exts/omni.ext.mobility_gen` path as a fallback
# so the script can be executed directly for debugging.
try:
    from omni.ext.mobility_gen.utils.global_utils import get_world
    from omni.ext.mobility_gen.writer import Writer
    from omni.ext.mobility_gen.reader import Reader
    from omni.ext.mobility_gen.build import load_scenario
except ModuleNotFoundError:
    import sys
    import os
    script_dir = os.path.dirname(__file__)
    candidate = os.path.abspath(os.path.join(script_dir, "..", "exts", "omni.ext.mobility_gen"))
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
    try:
        from omni.ext.mobility_gen.utils.global_utils import get_world
        from omni.ext.mobility_gen.writer import Writer
        from omni.ext.mobility_gen.reader import Reader
        from omni.ext.mobility_gen.build import load_scenario
    except Exception as e:
        print(f"[Replay] Failed importing omni.ext.mobility_gen after adding {candidate} to sys.path: {e}")
        raise


if __name__ == "__main__":

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', '0'):
            return False
        return bool(v)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--rgb_enabled", type=_str2bool, default=True)
    parser.add_argument("--segmentation_enabled", type=_str2bool, default=True)
    parser.add_argument("--depth_enabled", type=_str2bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=_str2bool, default=True)
    parser.add_argument("--normals_enabled", type=_str2bool, default=False)
    parser.add_argument("--lidar_enabled", type=_str2bool, default=True)
    parser.add_argument("--render_rt_subframes", type=int, default=1)
    parser.add_argument("--render_interval", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="Limit number of replay iterations (diagnostic)")

    args, unknown = parser.parse_known_args()

    import time as _time


    print(f"[SETUP] load_scenario ...", flush=True)
    _t0 = _time.time()
    try:
        scenario = load_scenario(os.path.join(args.input_path))
        print(f"[SETUP] load_scenario done ({_time.time()-_t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[ERROR] load_scenario failed: {e}", flush=True)
        raise

    print(f"[SETUP] get_world ...", flush=True)
    _t0 = _time.time()
    world = get_world()
    print("[DEBUG] Entrando no loop de replay...", flush=True)
    for i, step in enumerate(tqdm.tqdm(range(0, num_steps, args.render_interval))):
        if i == 0:
            print(f"[DEBUG] Primeira iteração do replay: i={i}, step={step}", flush=True)
        try:
            if args.max_steps is not None and i >= args.max_steps:
                break
            _loop_t0 = _time.time()
            print(f"[DEBUG] Loop {i}, step {step}: reading state_dict ...")
            state_dict = reader.read_state_dict(index=step)
            print(f"[DEBUG] Loop {i}, step {step}: loading state_dict ...")
            scenario.load_state_dict(state_dict)
            print(f"[DEBUG] Loop {i}, step {step}: writing replay data ...")
            scenario.write_replay_data()
            print(f"[DEBUG] Loop {i}, step {step}: simulation_app.update (START)")
            try:
                simulation_app.update()
                print(f"[DEBUG] Loop {i}, step {step}: simulation_app.update (DONE)")
            except Exception as e:
                print(f"[ERROR] Loop {i}, step {step}: simulation_app.update() EXCEPTION: {e}")
                raise
            print(f"[DEBUG] Loop {i}, step {step}: rep.orchestrator.step (START)")
            try:
                rep.orchestrator.step(
                    rt_subframes=args.render_rt_subframes,
                    delta_time=0.0,
                    pause_timeline=False,
                )
                print(f"[DEBUG] Loop {i}, step {step}: rep.orchestrator.step (DONE)")
            except Exception as e:
                print(f"[ERROR] Loop {i}, step {step}: rep.orchestrator.step() EXCEPTION: {e}")
                raise
            print(f"[LOOP] step {step} (iter {i}): render done ({_time.time()-_loop_t0:.2f}s)")
            print(f"[DEBUG] Loop {i}, step {step}: scenario.update_state ...")
            scenario.update_state()
            print(f"[DEBUG] Loop {i}, step {step}: scenario.update_state done")
            print(f"[DEBUG] Loop {i}, step {step}: collecting state outputs ...")
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
            print(f"[DEBUG] Loop {i}, step {step}: outputs written")
        except Exception as e:
            print(f"[ERROR] Exceção na iteração {i}, step {step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    print(f"[SETUP] world.reset ...", flush=True)
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
        print(f"[SETUP] enable_depth_rendering done ({_time.time()-_t0:.1f}s)")

    if args.instance_id_segmentation_enabled:
        print(f"[SETUP] enable_instance_id_segmentation_rendering ...")
        _t0 = _time.time()
        scenario.enable_instance_id_segmentation_rendering()
        print(f"[SETUP] enable_instance_id_segmentation_rendering done ({_time.time()-_t0:.1f}s)")

    if args.normals_enabled:
        print(f"[SETUP] enable_normals_rendering ...")
        _t0 = _time.time()
        scenario.enable_normals_rendering()
        print(f"[SETUP] enable_normals_rendering done ({_time.time()-_t0:.1f}s)")

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

        print(f"[DEBUG] Loop {i}, step {step}: reading state_dict ...")
        state_dict = reader.read_state_dict(index=step)
        print(f"[DEBUG] Loop {i}, step {step}: loading state_dict ...")
        scenario.load_state_dict(state_dict)
        print(f"[DEBUG] Loop {i}, step {step}: writing replay data ...")
        scenario.write_replay_data()

        print(f"[DEBUG] Loop {i}, step {step}: simulation_app.update (START)")
        try:
            simulation_app.update()
            print(f"[DEBUG] Loop {i}, step {step}: simulation_app.update (DONE)")
        except Exception as e:
            print(f"[ERROR] Loop {i}, step {step}: simulation_app.update() EXCEPTION: {e}")
            raise

        print(f"[DEBUG] Loop {i}, step {step}: rep.orchestrator.step (START)")
        try:
            rep.orchestrator.step(
                rt_subframes=args.render_rt_subframes,
                delta_time=0.0,
                pause_timeline=False,
            )
            print(f"[DEBUG] Loop {i}, step {step}: rep.orchestrator.step (DONE)")
        except Exception as e:
            print(f"[ERROR] Loop {i}, step {step}: rep.orchestrator.step() EXCEPTION: {e}")
            raise

        print(f"[LOOP] step {step} (iter {i}): render done ({_time.time()-_loop_t0:.2f}s)")

        print(f"[DEBUG] Loop {i}, step {step}: scenario.update_state ...")
        scenario.update_state()
        print(f"[DEBUG] Loop {i}, step {step}: scenario.update_state done")

        print(f"[DEBUG] Loop {i}, step {step}: collecting state outputs ...")
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
        print(f"[DEBUG] Loop {i}, step {step}: outputs written")

    print("[DONE] Replay finished successfully.")
    simulation_app.close()
    # Force exit: simulation_app.close() can hang during shutdown
    # (X connection broken, GPU resource cleanup deadlock, etc.)
    import os as _os
    _os._exit(0)