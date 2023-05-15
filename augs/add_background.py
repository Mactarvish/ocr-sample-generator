import os
import glob
import random

import numpy as np
import cv2

import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmenters import meta

        
class AddBackground(meta.Augmenter):
    """Blur an image by computing median values over neighbourhoods.

    Median blurring can be used to remove small dirt from images.
    At larger kernel sizes, its effects have some similarity with Superpixels.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    k : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size.

            * If a single ``int``, then that value will be used for the
              height and width of the kernel. Must be an odd value.
            * If a tuple of two ints ``(a, b)``, then the kernel size will be
              an odd value sampled from the interval ``[a..b]``. ``a`` and
              ``b`` must both be odd values.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the kernel size for the nth image. Expected to be discrete. If
              a sampled value is not odd, then that value will be increased
              by ``1``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AddBackground(k=5)

    Blur all images using a kernel size of ``5x5``.

    >>> aug = iaa.AddBackground(k=(3, 7))

    Blur images using varying kernel sizes, which are sampled uniformly from
    the interval ``[3..7]``. Only odd values will be sampled, i.e. ``3``
    or ``5`` or ``7``.

    """

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
        nb_images = len(images)
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