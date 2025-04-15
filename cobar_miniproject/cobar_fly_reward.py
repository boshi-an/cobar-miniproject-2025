import numpy as np
import gymnasium as gym
from flygym import Simulation
from flygym.arena import BaseArena
from cobar_miniproject.cobar_fly import CobarFly

class CobarFlyReward(CobarFly):
    
    def get_reward(self, sim: Simulation, obs: dict):

        odor_source_posision = sim.arena.odor_source
        self_position = sim.physics.bind(self._body_sensors[0]).sensordata.copy()
        self_velocity = sim.physics.bind(self._body_sensors[1]).sensordata.copy()
        self_orientation = sim.physics.bind(self._body_sensors[4]).sensordata.copy()

        # Copmute the distance to the odor source
        odor_source_position = np.array([odor_source_posision[0, 0], odor_source_posision[0, 1], 0])
        self_position = np.array([self_position[0], self_position[1], 0])
        vector_to_odor_source = odor_source_position - self_position
        vector_to_odor_source_normalized = vector_to_odor_source / (np.linalg.norm(vector_to_odor_source) + 1e-9)
        # Compute the reward for facing odor source
        facing_reward = np.dot(self_orientation, vector_to_odor_source_normalized)
        moving_reward = np.dot(self_velocity, vector_to_odor_source_normalized)
        distance_reward = 5 / (5 + np.linalg.norm(vector_to_odor_source))

        reward = facing_reward + moving_reward + distance_reward
        reward_dict = {
            "facing_reward": facing_reward,
            "moving_reward": moving_reward,
            "distance_reward": distance_reward,
        }

        return reward, reward_dict

    def get_observation(self, sim):
        obs = super().get_observation(sim)
        obs["heading"] = np.array(obs["heading"]).reshape((1,))
        obs["joints"] = obs["joints"].reshape(-1)
        obs["end_effectors"] = obs["end_effectors"].reshape(-1)
        obs["contact_forces"] = obs["contact_forces"].reshape(-1)
        obs["velocity"] = np.array(obs["velocity"]).reshape((2,))
        obs["odor_intensity"] = np.array(obs["odor_intensity"]).reshape((-1,))
        return obs
        
    def post_step(self, sim: Simulation):
        obs = self.get_observation(sim)
        reward, reward_dict = self.get_reward(sim, obs)
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()
        info["reward"] = reward_dict

        if self.enable_vision:
            vision_updated_this_step = sim.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        # Fly has flipped if the z component of the "up" cardinal vector is negative
        cardinal_vector_z = sim.physics.bind(self._body_sensors[6]).sensordata.copy()
        info["flip"] = cardinal_vector_z[2] < 0

        if self.head_stabilization_model is not None:
            # this is tracked to decide neck actuation for the next step
            self._last_observation = obs
            info["neck_actuation"] = self._last_neck_actuation

        
        return obs, reward, terminated, truncated, info

    def _define_observation_space(self, arena: BaseArena):
        _observation_space = {
            "joints": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3 * len(self.actuated_joints), )
            ),
            "end_effectors": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6 * 2, )),
            "contact_forces": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.contact_sensor_placements) * 3,)
            ),
            "heading": gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(1,)
            ),
            "velocity": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,)
            ),
        }
        if self.enable_olfaction:
            _observation_space["odor_intensity"] = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(2 * len(self._antennae_sensors), ),
            )
        if self.enable_vision:
            _observation_space["vision"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(2, self.config["vision"]["num_ommatidia_per_eye"], 2),
            )
            if self.render_raw_vision:
                _observation_space["raw_vision"] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, self.config["vision"]["raw_img_height_px"], self.config["vision"]["raw_img_width_px"], 3),
                )
        return gym.spaces.Dict(_observation_space)
