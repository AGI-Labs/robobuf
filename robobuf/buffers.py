import cv2, deepcopy
import numpy as np


class Transition:
    def __init__(self, obs, action, reward, prev_trans=None, next_trans=None):
        self.obs = obs
        self.action = action
        self.reward = reward

        self.prev_trans = prev_trans
        self.next_trans = next_trans

    @property
    def done(self):
        return self.next_trans is None

    @property
    def first(self):
        return self.prev_trans is None


class ObsWrapper:
    def __init__(self, obs, H=256, W=256):
        self.encoded_imgs = []
        self.H = H
        self.W = W

        num_img_keys = len([k for k in obs.keys() if "image" in k])
        for i in range(num_img_keys):
            self.encoded_imgs.append(obs[f"image_{i}"])

        # delete images from obs
        self.obs = {k: v for k, v in obs.items() if "image" not in k}
    

    def add_image(self, image):
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            self.encoded_imgs.append(encoded_image)
        else:
            print("failed to encode image")

    def image(self, idx=0):
        encoded_image_np = np.frombuffer(self.encoded_imgs[idx], dtype=np.uint8)
        cv2_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
        cv2_image = cv2.resize(cv2_image, (self.H, self.W))
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return rgb_image
    
    def to_dict(self):
        obs = deepcopy.copy(self.obs)
        for i in range(len(self.encoded_imgs)):
            obs[f"enc_image_{i}"] = self.encoded_imgs[i]
        return obs

    @classmethod
    def from_dict(self):  # this is dict to ObsWrapper parser
        raise NotImplementedError

class ReplayBuffer:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._buffer = []
        self._traj_starts = [] # first observation in each traj

    def add(self, transition: Transition):
        if len(self._buffer) >= self._max_size:
            print(f"buffer full with size {len(self._buffer)}")
            print(f"skip adding obs {transition}")
        self._buffer.append(transition)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(
        self,
        idx: int,
    ):
        return self._buffer[idx]

    def traj_starts(self):
        # returns iterator of trajectory starts?
        return self._traj_starts
    
    def to_traj_list(self):
        raise NotImplementedError()

    @classmethod
    def load_traj_list(self):  # this is traj_list --> buffer parser
        raise NotImplementedError()
