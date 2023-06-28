import numpy as np
import cv2

class Transition:
    def __init__(self, obs, action, next_obs, reward, done):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.reward = reward
        self.done = done

        self.prev = None
        self.next = None

    def get_data(self):
        return self.obs.get_data(), self.action, self.next_obs.get_data(), self.reward, self.done

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
    
    def get_data(self):
        obs = self.obs.copy()
        for i in range(len(self.encoded_imgs)):
            obs[f"image_{i}"] = self.encoded_imgs[i]
        return obs

class ReplayBuffer:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._buffer = []

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
    
    def get_data(self):
        return [t.get_data() for t in self._buffer]