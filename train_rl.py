from pathlib import Path
import importlib
import argparse
import sys
from tqdm import trange
from flygym import Camera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from flygym import Camera, SingleFlySimulation
from flygym.arena import FlatTerrain

from cobar_miniproject.cobar_fly_reward import CobarFlyReward
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from cobar_miniproject.fly_vec_env import FlyGymEnv

def run_simulation(
    level,
    seed,
    debug,
    max_steps,
    num_envs,
    train_steps,
    output_dir="outputs",
    progress=True,
    device="cpu",
):
    timestep = 1e-4

    print("Using parameters:")
    print(f"Level: {level}")
    print(f"Seed: {seed}")
    print(f"Debug: {debug}")
    print(f"Max Steps: {max_steps}")
    print(f"Number of Envs: {num_envs}")
    print(f"Train Steps: {train_steps}")
    print(f"Output Directory: {output_dir}")

    def get_env() :

        fly = CobarFlyReward(
            debug=debug,
            enable_vision=True,
            render_raw_vision=True,
        )

        if level <= -1:
            level_arena = FlatTerrain()
        elif level <= 1:
            # levels 0 and 1 don't need the timestep
            level_arena = levels[level](fly=fly, seed=seed)
        else:
            # levels 2-4 need the timestep
            level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)
        
        cam_params = {"pos": (0, 0, 80)}

        # cam = Camera(
        #     attachment_point=level_arena.root_element.worldbody,
        #     camera_name="camera_top_zoomout",
        #     targeted_fly_names=[fly.name],
        #     camera_parameters=cam_params,
        #     play_speed=0.2,
        # )

        sim = SingleFlySimulation(
            fly=fly,
            cameras=None,
            timestep=timestep,
            arena=level_arena,
        )

        gym_env = FlyGymEnv(sim, max_steps)
        
        return gym_env

    # Create the environment
    # env = SubprocVecEnv([get_env for _ in range(num_envs)], start_method="fork")
    env =  get_env()

    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, device=device)

    # Train the model
    model.learn(total_timesteps=train_steps)

    # Save the trained model
    model.save(f"{output_dir}/ppo_fly_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
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
        default=256,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of parallel simulations.",
        default=4,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        help="Number of steps to train the network.",
        default=10000,
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
        "--device",
        type=str,
        default="cpu",
        help="Device used for training.",
    )
    args = parser.parse_args()

    run_simulation(
        level=args.level,
        seed=args.seed,
        debug=args.debug,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_envs=args.num_envs,
        train_steps=args.train_steps,
        progress=args.progress,
        device=args.device
    )
