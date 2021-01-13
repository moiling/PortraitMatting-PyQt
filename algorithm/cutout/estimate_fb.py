import numpy as np


def vec_vec_outer(a, b):
    return np.einsum("...i,...j", a, b)


def inv2(mat):
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]

    inv_det = 1 / (a * d - b * c)

    inv = np.empty(mat.shape)

    inv[..., 0, 0] = inv_det * d
    inv[..., 0, 1] = inv_det * -b
    inv[..., 1, 0] = inv_det * -c
    inv[..., 1, 1] = inv_det * a

    return inv


def pixel_coordinates(w, h, flat=False):
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    if flat:
        x = x.flatten()
        y = y.flatten()

    return x, y


def resize_nearest(image, new_width, new_height):
    old_height, old_width = image.shape[:2]

    x = np.arange(new_width)[np.newaxis, :]
    y = np.arange(new_height)[:, np.newaxis]
    x = x * old_width / new_width
    y = y * old_height / new_height
    x = np.clip(x.astype(np.int32), 0, old_width - 1)
    y = np.clip(y.astype(np.int32), 0, old_height - 1)

    if len(image.shape) == 3:
        image = image.reshape(-1, image.shape[2])
    else:
        image = image.ravel()

    return image[x + y * old_width]


def estimate_foreground_background(
        input_image,
        input_alpha,
        min_size=2,
        growth_factor=2,
        regularization=1e-5,
        n_iter_func=lambda w, h: 5 if max(w, h) <= 64 else 1,
        print_info=False,
):
    """
    Estimate cutout and background of an image using a multilevel
    approach.
    min_size: int > 0
        Minimum image size at which to start solving.
    growth_factor: float64 > 1.0
        Image size is increased by growth_factor each level.
    regularization: float64
        Smoothing factor for undefined cutout/background regions.
    n_iter_func: func(width: int, height: int) -> int
        How many iterations to perform at a given image size.
    print_info:
        Wheter to print debug information during iterations.
    Returns
    -------
    F: np.ndarray of dtype np.float64
        Foreground image.
    B: np.ndarray of dtype np.float64
        Background image.
    """

    assert (min_size >= 1)
    assert (growth_factor > 1.0)
    h0, w0 = input_image.shape[:2]

    if print_info:
        print("Solving for cutout and background using multilevel method")

    # Find initial image size.
    if w0 < h0:
        w = min_size
        # ceil rounding one level faster sometimes
        h = int(np.ceil(min_size * h0 / w0))
    else:
        w = int(np.ceil(min_size * w0 / h0))
        h = min_size

    if print_info:
        print("Initial size: %d-by-%d" % (w, h))

    # Generate initial cutout and background from input image
    F = resize_nearest(input_image, w, h)
    B = F.copy()

    while True:
        if print_info:
            print("New level of size: %d-by-%d" % (w, h))

        # Resize image and alpha to size of current level
        image = resize_nearest(input_image, w, h)
        alpha = resize_nearest(input_alpha, w, h)

        # Iterate a few times
        n_iter = n_iter_func(w, h)
        for iteration in range(n_iter):
            if print_info:
                print("Iteration %d of %d" % (iteration + 1, n_iter))

            x, y = pixel_coordinates(w, h, flat=True)

            # Make alpha into a vector
            if len(alpha.shape) == 3:
                a = alpha[:, :, 0].reshape(w * h)
            else:
                a = alpha.reshape(w * h)

            # Build system of linear equations
            U = np.stack([a, 1 - a], axis=1)
            A = vec_vec_outer(U, U)
            b = vec_vec_outer(U, image.reshape(w * h, 3))

            # For each neighbor
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x2 = np.clip(x + dx, 0, w - 1)
                y2 = np.clip(y + dy, 0, h - 1)

                # Vectorized neighbor coordinates
                j = x2 + y2 * w

                # Gradient of alpha
                da = regularization + np.abs(a - a[j])

                # Update matrix of linear equation system
                A[:, 0, 0] += da
                A[:, 1, 1] += da

                # Update rhs of linear equation system
                b[:, 0, :] += da.reshape(w * h, 1) * F.reshape(w * h, 3)[j]
                b[:, 1, :] += da.reshape(w * h, 1) * B.reshape(w * h, 3)[j]

            # Solve linear equation system for cutout and background
            fb = np.clip(np.matmul(inv2(A), b), 0, 1)

            F = fb[:, 0, :].reshape(h, w, 3)
            B = fb[:, 1, :].reshape(h, w, 3)

        # If original image size is reached, return result
        if w >= w0 and h >= h0:
            return F, B

        # Grow image size to next level
        w = min(w0, int(np.ceil(w * growth_factor)))
        h = min(h0, int(np.ceil(h * growth_factor)))

        F = resize_nearest(F, w, h)
        B = resize_nearest(B, w, h)