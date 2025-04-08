import numpy as np
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
import gymnasium as gym
from cobar_miniproject.base_controller import BaseController

# Initialize CPG network
intrinsic_freqs = np.ones(6) * 12 * 1.5
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20


class Controller2D():
    def __init__(
        self,
        timestep: float,
        seed: int = 0,
        leg_step_time=0.025,
    ):
        """Controller that listens to your keypresses and uses these to
        modulate CPGs that control fly walking and turning.

        Parameters
        ----------
        timestep : float
            Timestep of the simulation.
        seed : int
            Random seed.
        leg_step_time : float, optional
            Duration of each step, by default 0.025.
        """
        self.timestep = timestep
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep,
            intrinsic_freqs,
            intrinsic_amps,
            coupling_weights,
            phase_biases,
            convergence_coefs,
            np.random.rand(6),
            np.random.rand(6),
            seed,
        )

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)
        self.phase_increment = self.timestep / leg_step_time * 2 * np.pi

        self.quit = False

    def get_cpg_joint_angles(self, command):
        # action = np.array([self.gain_left, self.gain_right])

        amps = np.repeat(np.abs(command[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if command[0] > 0 else -1
        freqs[3:] *= 1 if command[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        # TODO: update this following https://github.com/NeLy-EPFL/flygym/blob/main/flygym/examples/locomotion/turning_controller.py
        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

    def get_actions(self, command):
        return self.get_cpg_joint_angles(command)

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)

    @property
    def command_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # Example action space
    
    # @property
    # def action_space(self):
    #     return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # Example action space
