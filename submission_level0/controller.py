import os
import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .odor import OdorNavigator
from .vision import VisualNavigator
from .detector import BallDetector
from .integrator import PathIntegrator
from cobar_miniproject.lowlevel_controller import Controller2D

class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
        speed_scale=1.0
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.track = True
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.speed_scale = speed_scale
        self.path = os.path.dirname(os.path.realpath(__file__))
        
        self.odor_navigator = OdorNavigator(history_length=64)
        self.visual_navigator = VisualNavigator()
        self.ball_detector = BallDetector(os.path.join(self.path, "data/model_epoch_20.pth"))
        self.path_integrator = PathIntegrator()
        self.lowlevel_controller = Controller2D(timestep=timestep, seed=seed, leg_step_time=0.005)
        self.timestep = timestep

    def get_actions(self, obs: Observation) -> Action:
        odor_angle = self.odor_navigator.get_odor_angle(obs["odor_intensity"])
        visual_override, visual_cmd = self.visual_navigator.get_obstacle_pos(
            obs["vision"],
            obs.get("raw_vision", None),
            obs.get("vision_updated", True),
        )
        avoid_override, avoid_ball_cmd = self.ball_detector.get_ball_pos(
            obs["vision"],
            obs.get("raw_vision", None),
            obs.get("vision_updated", True),
        )
        nav_cmd = self.path_integrator.get_nav_command_from_obs(obs, dt=self.timestep)
        # Priority:
        # avaoid_ball > visual > odor
        
        odor_high_level_action = np.array([self.speed_scale * (1 + min(odor_angle, 0)), self.speed_scale * (1 - max(odor_angle, 0))])
        visual_high_level_action = np.array([self.speed_scale * visual_cmd[0], self.speed_scale * visual_cmd[1]])
        avoid_ball_high_level_action = np.array([self.speed_scale * avoid_ball_cmd[0], self.speed_scale * avoid_ball_cmd[1]])
        path_integrate_high_level_action = np.array([self.speed_scale * nav_cmd[0], self.speed_scale * nav_cmd[1]])

        if obs["reached_odour"] and self.track:
            print("reached")
            self.track = False
            self.path_integrator.return_status(obs)
        if obs["reached_odour"]:
            high_level_action = path_integrate_high_level_action
            # print(high_level_action)
            if self.path_integrator.returned_home:
                self.quit = True
        elif avoid_override :
            high_level_action = avoid_ball_high_level_action
        elif visual_override :
            high_level_action = visual_high_level_action
        else :
            high_level_action = odor_high_level_action

        self.odor_navigator.update(high_level_action)
        action = self.lowlevel_controller.get_actions(high_level_action)
        
        return action

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
