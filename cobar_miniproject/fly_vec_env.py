# Use stable baselines 3 to train a PPO controller
# Wrap the simulation in a gym-like environment
import gymnasium as gym
from cobar_miniproject.lowlevel_controller import Controller2D
from flygym import Simulation

class FlyGymEnv(gym.Env):
    def __init__(self, sim: Simulation, max_steps):
        self.sim = sim
        self.max_steps = max_steps
        self.lowlevel_controller = Controller2D(sim.timestep, seed=0)
        assert self.lowlevel_controller.command_space == self.action_space, "Controller and simulation action spaces do not match."

    def render(self, mode="human"):
        if mode == "human":
            self.sim.render()
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")
    
    def reset(self, seed=0):
        self.current_step = 0
        return self.sim.reset()

    def step(self, action):
        controller_output = self.lowlevel_controller.get_actions(action)
        obs, reward, terminated, truncated, info = self.sim.step(controller_output)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        self.sim.render()

    def close(self):
        self.sim.close()

    @property
    def observation_space(self):
        return self.sim.observation_space

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # Example action space
