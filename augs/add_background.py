import os
import glob
import random

import numpy as np
import cv2

import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmenters import meta

        
class AddBackground(meta.Augmenter):
    def __init__(self, background_dir, alpha,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddBackground, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.background_dir = background_dir
        self.bg_image_paths = glob.glob(os.path.join(self.background_dir, "*.jpg")) + \
                                glob.glob(os.path.join(self.background_dir, "*.png"))
        assert len(self.bg_image_paths) != 0, self.background_dir
        assert isinstance(alpha, tuple) and len(alpha) == 2 and alpha[0] >= 0 and alpha[1] >= alpha[0] and alpha[1] <= 1, alpha
        self.alpha = alpha

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        for i, image in enumerate(images):
            src_bg_path = random.sample(self.bg_image_paths, 1)[0]
            src_bg_np = cv2.imread(src_bg_path)
            src_bg_np = cv2.resize(src_bg_np, (image.shape[1], image.shape[0]))
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            mixed_np = np.uint8(src_bg_np.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha))
            batch.images[i] = mixed_np

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.alpha, self.background_dir]