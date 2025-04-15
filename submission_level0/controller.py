import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .odor import OdorNavigator
from .vision import VisualNavigator
from cobar_miniproject.lowlevel_controller import Controller2D

class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
        speed_scale=1.2
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.speed_scale = speed_scale
        
        self.odor_navigator = OdorNavigator(history_length=64)
        self.visual_navigator = VisualNavigator()
        self.lowlevel_controller = Controller2D(timestep=timestep, seed=seed, leg_step_time=0.01)

    def get_actions(self, obs: Observation) -> Action:
        odor_angle = self.odor_navigator.get_odor_angle(obs["odor_intensity"])
        visual_cmd = self.visual_navigator.get_obstacle_pos(obs["vision"], obs.get("vision_updated", True))
        
        if visual_cmd[0] != 0 or visual_cmd[1] != 0:
            high_level_action = np.array([self.speed_scale * visual_cmd[0], self.speed_scale * visual_cmd[1]])
        else :
            high_level_action = np.array([self.speed_scale * (1 + min(odor_angle, 0)), self.speed_scale * (1 - max(odor_angle, 0))])
        self.odor_navigator.update(high_level_action)
        action = self.lowlevel_controller.get_actions(high_level_action)
        
        return action

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
