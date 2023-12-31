import copy

import cv2
import numpy as np

class Transition:
    def __init__(self, obs, action, reward, prev_trans=None, next_trans=None):
        self.obs = obs
        self.action = action
        self.reward = reward

        self.prev = prev_trans
        self.next = next_trans

    @property
    def done(self):
        return self.next is None

    @property
    def first(self):
        return self.prev is None

    def to_tuple(self):
        return (self.obs.to_dict(), self.action, self.reward)


class ObsWrapper:
    def __init__(self, obs, H=256, W=256, skip_cam_idxs=[]):
        self.H = H
        self.W = W

        self.obs = {k: v for k, v in obs.items() if "cam" not in k}

        cam_keys = [k for k in obs.keys() if "cam" in k]
        for i, cam_key in enumerate(cam_keys):
            if i not in skip_cam_idxs:
                if ("enc" not in cam_key):
                    self.obs[f"enc_cam_{i}"] = self._encode_image(obs[cam_key])
                else:
                    self.obs[cam_key] = obs[cam_key]
            
    @property
    def state(self):
        return self.obs["state"]

    def _encode_image(self, rgb_image):
        bgr_image = rgb_image[:,:,::-1]
        if bgr_image.shape[:2] != (self.H, self.W):
            bgr_image = cv2.resize(bgr_image, (self.H, self.W), 
                                   interpolation=cv2.INTER_CUBIC)
        _, encoded = cv2.imencode(".jpg", bgr_image)
        return encoded

    def image(self, idx=0):
        encoded_image_np = np.frombuffer(self.obs[f"enc_cam_{idx}"], dtype=np.uint8)
        bgr_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
        assert bgr_image.shape[:2] == (self.H, self.W), "height and width don't match!"
        rgb_image = bgr_image[:,:,::-1]
        return rgb_image

    def to_dict(self):
        obs = copy.deepcopy(self.obs)
        return obs

    @classmethod
    def from_dict(cls, obs):  # this is dict to ObsWrapper parser
        return cls(obs)


class ReplayBuffer:
    def __init__(self, max_size: int = np.inf):
        self._max_size = max_size
        self._buffer = []
        self._traj_starts = []  # first observation in each traj

    def add(self, transition: Transition, is_first=False):
        if len(self._buffer) >= self._max_size:
            print(f"buffer full with size {len(self._buffer)}")
            print(f"skip adding obs {transition}")

        if not is_first:
            transition.prev = self._buffer[-1] if len(self._buffer) > 0 else None
            if len(self._buffer) > 0:
                self._buffer[-1].next = transition

        self._buffer.append(transition)

        if is_first:
            self._traj_starts.append(transition)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(
        self,
        idx: int,
    ):
        return self._buffer[idx]

    def clear(self):
        self._buffer.clear()
        self._traj_starts.clear()

    def traj_starts(self):
        return self._traj_starts

    def to_traj_list(self):
        trajs = []
        for traj_start in self._traj_starts:
            # iterate through the transitions till the end of the trajectory
            traj = []
            transition = traj_start
            while not transition.done:
                traj.append(transition.to_tuple())
                transition = transition.next

            # append last transition
            traj.append(transition.to_tuple())

            trajs.append(traj)
        return trajs

    @classmethod
    def load_traj_list(cls, traj_list):  # this is traj_list --> buffer parser
        buffer = cls()
        buffer.append_traj_list(traj_list)
        return buffer

    def append_traj_list(self, traj_list, skip_actors=["ai_agent"]):
        for traj in traj_list:
            trans_count = 0
            # print(f"adding new traj with {len(traj)} transitions")
            for i, trans in enumerate(traj):
                obs, action, reward = trans
                if "actor" in obs.keys() and obs["actor"] in skip_actors:
                    continue                    
                transition = Transition(ObsWrapper.from_dict(obs), action, reward)
                self.add(transition, trans_count == 0)
                trans_count += 1