from pathlib import Path
import importlib
import argparse
import sys
import numpy as np
from tqdm import trange
from flygym import Camera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from flygym import Camera, SingleFlySimulation
from flygym.arena import FlatTerrain
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from data_saver import DataSaver
import cv2

WITH_FLY_VISION = 1
WITH_RAW_VISION = 2

VISUALISATION_MODE = WITH_FLY_VISION

def run_simulation(
    submission_dir,
    level,
    seed,
    debug,
    max_steps,
    output_dir="outputs",
    progress=True,
    show_viewer=False,
    collect_data=False
):
    sys.path.append(str(submission_dir.parent))
    module = importlib.import_module(submission_dir.name)
    if collect_data :
        # Create data collection directory
        file_id = seed
        data_dir = Path(output_dir) / "data" / str(file_id)
        while data_dir.exists():
            file_id += 1
            data_dir = Path(output_dir) / "data" / str(file_id)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_saver = DataSaver(data_dir)

    controller = module.controller.Controller()
    timestep = 1e-4

    fly = CobarFly(
        debug=debug,
        enable_vision=True,
        render_raw_vision=True,
    )

    other_kwargs = {}
    if collect_data :
        other_kwargs["looming_lambda"] = 3.0

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed, **other_kwargs)
    
    cam_params = {"pos": (0, 0, 80)}

    cam = Camera(
        attachment_point=level_arena.root_element.worldbody,
        camera_name="camera_top_zoomout",
        targeted_fly_names=[fly.name],
        camera_parameters=cam_params,
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    # run cpg simulation
    obs, info = sim.reset()

    if progress:
        step_range = trange(max_steps)
    else:
        step_range = range(max_steps)
        
    # create window
    if show_viewer:
        cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    try: 
        for i in step_range:
            # Get observations
            obs, reward, terminated, truncated, info = sim.step(controller.get_actions(obs))
            rendered_img = sim.render()[0]
            
            if show_viewer :
                if rendered_img is not None:
                    
                    if VISUALISATION_MODE == WITH_FLY_VISION:
                        rendered_img = render_image_with_vision(
                            rendered_img, get_fly_vision(fly), obs["odor_intensity"],
                        )
                    elif VISUALISATION_MODE == WITH_RAW_VISION:
                        rendered_img = render_image_with_vision(
                            rendered_img, get_fly_vision_raw(fly), obs["odor_intensity"],
                        )
                    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Simulation", rendered_img)
                    cv2.waitKey(1)
            
            if controller.done_level(obs):
                # finish the path integration level
                break
                
            obs_ = obs.copy()
            if collect_data and obs["vision_updated"]:
                mask, mask_hex = controller.visual_navigator._get_raw_vision_mask(obs["raw_vision"])
                human_readable = controller.visual_navigator.get_human_readable(obs["vision"])
                human_readable = np.stack(human_readable, axis=0)
                # Save raw_vision, mask and mask_hex to data_dir
                data_saver.append(
                    {
                        "mask": mask,
                        "mask_hex": mask_hex,
                        "raw_vision": obs["raw_vision"],
                        "human_readable": human_readable,
                        "vision_hex": obs["vision"],
                    }
                )


            if not obs_["vision_updated"]:
                if "vision" in obs_:
                    del obs_["vision"]
                if "raw_vision" in obs_:
                    del obs_["raw_vision"]

            if hasattr(controller, "quit") and controller.quit:
                print("Simulation terminated by user.")
                break
            if hasattr(level_arena, "quit") and level_arena.quit:
                print("Target reached. Simulation terminated.")
                break

    finally :
        # Save video
        save_path = Path(output_dir) / f"level{level}_seed{seed}.mp4"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cam.save_video(save_path, stabilization_time=0)
        print(f"Video saved to {save_path}")

        if collect_data:
            # Save data
            data_saver.close()
            print(f"Data saved to {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory containing the controller module.",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Simulation level to run (e.g., -1 for FlatTerrain, 0-4 for specific levels).",
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the simulation.",
        default=0,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to run the simulation.",
        default=100000,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the simulation.",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the simulation outputs (default: 'outputs').",
        default="outputs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during simulation.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Show viewer.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect visual data.",
    )
    args = parser.parse_args()

    run_simulation(
        submission_dir=args.submission_dir,
        level=args.level,
        seed=args.seed,
        debug=args.debug,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        progress=args.progress,
        show_viewer=args.view,
        collect_data=args.collect
    )
