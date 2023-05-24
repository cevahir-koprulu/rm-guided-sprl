import os
import copy
import numpy as np

import gym
from gym.envs.robotics import rotations, utils
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


DEFAULT_SIZE = 500


def get_distance(loc_a, loc_b):
    assert loc_a.shape == loc_b.shape
    return np.linalg.norm(loc_a - loc_b, axis=-1)


class ContextualFetchPushAndPlay(gym.Env):
    # Context Range: [-0.15, -0.5] x [0.5, 0.15]
    LOC_1 = np.array([-0.15, -0.15])
    LOC_2 = np.array([0.15, 0.15])

    def __init__(
        self,
        model_path,
        block_gripper,
        n_substeps,
        gripper_extra_height,
        target_offset,
        obj_range,
        distance_threshold,
        initial_qpos,
        context=None,
        product_cmdp=False,
        rm_state_onehot=True,
    ):
        """Initializes a new ContextualFetchPickAndPlay environment.

        Args:
            model_path (string): path to the environments XML file
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.distance_threshold = distance_threshold

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        state = self._get_obs()["observation"]
        n_actions = 4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=state.shape, dtype="float32")
        self.product_cmdp = product_cmdp
        self.rm_state_onehot = rm_state_onehot
        if context is None:
            context = np.concatenate([self.LOC_1, self.LOC_2])
        self.context = context
        self._update_env_config()
        self.dense_reward = True
        self.rm_info = {"num_states": 3,
                        "sink": None,
                        "goal": 2}
        self.rm_transition_context_map = {
            (0, 0): [0, 1],
            (0, 1): [0, 1],
            (1, 1): [2, 3],
            (1, 2): [2, 3],
        }
        
        if self.product_cmdp:
            if self.rm_state_onehot:
                low_ext = np.concatenate((self.observation_space.low, 
                                          np.zeros(self.rm_info["num_states"])))
                high_ext = np.concatenate((self.observation_space.high, 
                                           np.ones(self.rm_info["num_states"])))
            else:
                low_ext = np.concatenate((self.observation_space.low, 
                                          np.array(0.)))
                high_ext = np.concatenate((self.observation_space.high, 
                                           np.array(self.rm_info["num_states"]-1)))
            self.observation_space = spaces.Box(low=low_ext, 
                                                high=high_ext)

        self.rewards = {
            0: {0: None, 1: 10.},
            1: {1: None, 2: 100.},
            2: {2: None},
        }

        self._rm_state = None
        self._past_rm_state = None
        self._state = None
        self._num_step = 0

    def _update_env_config(self):
        self._locs = {
            "loc_1": None,
            "loc_2": None,
        }
        for loc_i, loc_key in enumerate(self._locs):
            augmented_c = np.concatenate([self.context[loc_i*2:(loc_i+1)*2], np.zeros(1)])
            self._locs[loc_key] = self.initial_gripper_xpos[:3] + augmented_c
            self._locs[loc_key] += self.target_offset + self.height_offset

    def get_rm_transition(self):
        return self._past_rm_state, self._rm_state

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self._num_step = 0
        self._past_rm_state = None
        self._rm_state = 0
        _state = self._get_obs()["observation"]
        if self.product_cmdp:
            if self.rm_state_onehot:
                _state_rm_ext = np.zeros(self.rm_info["num_states"])
                _state_rm_ext[self._rm_state] = 1.
                _state = np.concatenate((_state, _state_rm_ext))
            else:
                _state = np.concatenate((_state, np.array(self._rm_state)))
        self._state = np.copy(_state)
        return _state

    def step(self, action):
        self._update_env_config()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        next_obs = self._get_obs()
        next_state = next_obs["observation"]
        next_object_pos = next_obs["object_pos"]

        done = False
        if self._rm_state == 0:
            next_rm_state = 0
            d = get_distance(self._locs["loc_1"], next_object_pos)
            if d <= self.distance_threshold:
                next_rm_state = 1
        elif self._rm_state == 1:
            next_rm_state = 1
            d = get_distance(self._locs["loc_2"], next_object_pos)
            if d <= self.distance_threshold:
                next_rm_state = 2
                done = True
        elif self._rm_state == 2:
            next_rm_state = 2

        reward_rm = self.rewards[self._rm_state][next_rm_state]
        if reward_rm is None:
            reward_rm = 0
        if self.dense_reward:
            reward = -d + reward_rm # dense reward in fetch env
        else:
            reward = -1.0 + reward_rm # sparse reward in fetch env


        if self._rm_state != next_rm_state:
            print(f"Object is moved to new loc! from rm_state {self._rm_state} to next_rm_state {next_rm_state} with reward_rm+reward {reward_rm}+{reward-reward_rm}!")

        self._num_step += 1
        self._past_rm_state = copy.deepcopy(self._rm_state)
        self._rm_state = copy.deepcopy(next_rm_state)

        info = {}
        info["success"] = next_rm_state == self.rm_info["goal"]
        info["mission"] = next_rm_state

        if self.product_cmdp:
            if self.rm_state_onehot:
                _state_rm_ext = np.zeros(self.rm_info["num_states"])
                _state_rm_ext[self._rm_state] = 1.
                next_state = np.concatenate((next_state, _state_rm_ext))
            else:
                next_state = np.concatenate((next_state, np.array(self._rm_state)))
        self._state = np.copy(next_state)
        return next_state, reward, done, info

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # GoalEnv methods
    # ----------------------------

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)

        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos("object0")
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        # velocities
        object_velp = self.sim.data.get_site_xvelp("object0") * dt
        object_velr = self.sim.data.get_site_xvelr("object0") * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "object_pos": np.squeeze(object_pos.copy()),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self._locs["loc_2"] - sites_offset[0]
        self.sim.forward()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("object0")[2]


