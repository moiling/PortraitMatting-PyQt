import numpy as np
from .estimate_fb import estimate_foreground_background


def cutout(image, alpha):
    """
    :param   image: shape: [H, W, RGB(3) ] range: [0, 1]
    :param   alpha: shape: [H, w, 1      ] range: [0, 1]
    :return       : shape: [H, W, RGBA(4)] range: [0, 1]
    """
    fg, _ = estimate_foreground_background(image[..., ::-1], alpha)  # [H, W, BGR(3) ]
    cutout = np.zeros((image.shape[0], image.shape[1], 4))
    cutout[..., :3] = fg[..., ::-1]
    cutout[..., 3] = alpha.astype(np.float32).squeeze(axis=2)  # [H, W, RGBA(4)]
    return cutout