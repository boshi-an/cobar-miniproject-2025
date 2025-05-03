from re import M
import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .odor import OdorNavigator
from .vision import VisualNavigator
from .detector import BallDetector
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
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.speed_scale = speed_scale
        
        self.odor_navigator = OdorNavigator(history_length=64)
        self.visual_navigator = VisualNavigator()
        self.ball_detector = BallDetector("outputs/cnn/model_epoch_20.pth")
        self.lowlevel_controller = Controller2D(timestep=timestep, seed=seed, leg_step_time=0.005)

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
        # Priority:
        # avaoid_ball > visual > odor
        
        odor_high_level_action = np.array([self.speed_scale * (1 + min(odor_angle, 0)), self.speed_scale * (1 - max(odor_angle, 0))])
        visual_high_level_action = np.array([self.speed_scale * visual_cmd[0], self.speed_scale * visual_cmd[1]])
        avoid_ball_high_level_action = np.array([self.speed_scale * avoid_ball_cmd[0], self.speed_scale * avoid_ball_cmd[1]])

        if avoid_override :
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
