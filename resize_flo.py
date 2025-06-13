import numpy as np
from scipy.ndimage import zoom
import argparse

import utils

def resize_flat_vector_field(u_flat, v_flat, original_width, original_height,
                              new_width, new_height, order=1):
    """
    Resize a flattened 2D vector field defined on a uniform grid.

    Parameters:
    - u_flat, v_flat: 1D numpy arrays (flattened from row-major 2D u and v).
    - original_width, original_height: Dimensions of the original grid.
    - new_width, new_height: Target dimensions for resized field.
    - order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)

    Returns:
    - u_resized_flat, v_resized_flat: 1D flattened resized vector field components.
    """
    # Reshape the flattened arrays into 2D
    u_2d = u_flat.reshape((original_height, original_width))
    v_2d = v_flat.reshape((original_height, original_width))

    # Compute zoom factors
    zoom_y = new_height / original_height
    zoom_x = new_width / original_width

    # Interpolate each component
    u_resized = zoom(u_2d, (zoom_y, zoom_x), order=order)
    v_resized = zoom(v_2d, (zoom_y, zoom_x), order=order)

    # Flatten the resized arrays back to 1D
    u_resized_flat = u_resized.flatten()
    v_resized_flat = v_resized.flatten()

    return u_resized_flat, v_resized_flat

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("flo", help="flo file")
parser.add_argument("out", help="output")

args = parser.parse_args()

w, h, u, v = utils.openFlo(args.flo)

new_h, new_w = int(w/2), int(h/2)

u_resized, v_resized = resize_flat_vector_field(u, v, w, h, new_h, new_w, order=1)

utils.saveFlo(new_w, new_h, u_resized, v_resized, args.out)