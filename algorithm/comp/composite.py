
def composite(cutout, bg):
    """
    :param  cutout: shape: [H, W, RGBA(4)] range: [0, 1]
    :param  bg    : shape: [BGR(3)]        range: [0, 1]
    :return       : shape: [H, W, RGB(3) ] range: [0, 1]
    """
    alpha = cutout[:, :, 3:4]
    fg = cutout[:, :, :3]
    image = alpha * fg + (1 - alpha) * bg
    return image
