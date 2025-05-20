import numpy as np

class PathIntegrator:
    def __init__(self, start_pos=np.zeros(2)):
        self.pos = np.array(start_pos, dtype=np.float64)
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.return_threshold = 1
        self.returned_home = False

    def get_nav_command_from_obs(self, obs: dict, dt: float, Kv=1.0, Kh=2.0):
        velocity_xy = np.array(obs["velocity"][:2]).flatten()
        heading_angle = obs["heading"]  # rad

        # path integration
        R = np.array([
            [np.cos(heading_angle), -np.sin(heading_angle)],
            [np.sin(heading_angle),  np.cos(heading_angle)],
        ])
        world_vel = R @ velocity_xy
        self.pos += world_vel * dt

        # trace back control
        delta = self.start_pos - self.pos
        distance = np.linalg.norm(delta)

        # if reached the target
        if distance < self.return_threshold and obs["reached_odour"]:
            self.returned_home = True
            return np.array([0.0, 0.0], dtype=np.float32)

        # navigation
        theta_target = np.arctan2(delta[1], delta[0])
        angle_error = (theta_target - heading_angle + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

        # rescale
        v = min(Kv * distance, 1.0)       # Forward drive strength (capped at 1.0 to avoid overspeed)
        w = np.tanh(Kh * angle_error)     # Turning component using tanh to keep within (-1, 1)
        left_cmd = v - w
        right_cmd = v + w

        max_val = max(abs(left_cmd), abs(right_cmd), 1.0)  # 避免除0
        cmd = np.array([left_cmd / max_val, right_cmd / max_val], dtype=np.float32)

        return cmd
    
    def return_status(self,obs):
        print("========== [PathIntegrator Status] ==========")
        print(f"Start pos      : {self.start_pos}")
        print(f"Current pos    : {self.pos}")
        print(f"Returned home? : {self.returned_home}")
        print(f"Last velocity  : {obs["velocity"]}")
        print(f"Last heading   : {obs["heading"]}")
        print(f"Diatance       : {obs["heading"]}")
        print("=============================================")




