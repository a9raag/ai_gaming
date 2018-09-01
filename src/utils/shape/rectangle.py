import numpy as np
from collections import namedtuple

def rectangle_properties(result):
    top_left_x, top_left_y = result["topleft"]["x"], result["topleft"]["y"]
    btm_right_x, btm_right_y = result["bottomright"]["x"], result["bottomright"]["y"]
    top_right_x, top_right_y = result["bottomright"]["x"], result["topleft"]["y"]
    btm_left_x, btm_left_y = result["topleft"]["x"], result["bottomright"]["y"]
    top_left = np.array((top_left_x, top_left_y))
    btm_right = np.array((btm_right_x, btm_right_y))
    top_right = np.array((top_right_x, top_right_y))
    btm_left = np.array((btm_left_x, btm_left_y))
    length = np.linalg.norm(top_left - top_right)
    width = np.linalg.norm(top_left - btm_left)
    diagonal = np.linalg.norm(top_left - btm_right)
    Rectangle = namedtuple("Rectangle" , ["length", "width","diagonal"], verbose=False)
    return Rectangle(length, width, diagonal)


def compute_diagonal(result):
    top_left_x, top_left_y, btm_right_x, btm_right_y = result["topleft"]["x"], result["topleft"]["y"], \
                                                       result["bottomright"]["x"], result["bottomright"]["y"]
    top_left = np.array((top_left_x, top_left_y))
    btm_right = np.array((btm_right_x, btm_right_y))
    diagonal = np.linalg.norm(top_left - btm_right)
    return diagonal


def compute_width(result):
    # top_left_x, top_left_y = result["topleft"]["x"], result["topleft"]["y"]
    # # btm_right_x, btm_right_y = result["bottomright"]["x"], result["bottomright"]["y"]
    # top_right_x, top_right_y = result["bottomright"]["x"], result["topleft"]["y"]
    # btm_left_x, btm_left_y = result["topleft"]["x"], result["bottomright"]["y"]
    # top_left = np.array((top_left_x, top_left_y))
    # # btm_right = np.array((btm_right_x, btm_right_y))
    # top_right = np.array((top_right_x, top_right_y))
    # btm_left = np.array((btm_left_x, btm_left_y))
    # length = np.linalg.norm(top_left - top_right)
    # width = np.linalg.norm(top_left - btm_left)

    return rectangle_properties(result)["width"]


def compute_length(result):
    # top_left_x, top_left_y = result["topleft"]["x"], result["topleft"]["y"]
    # top_right_x, top_right_y = result["bottomright"]["x"], result["topleft"]["y"]
    # top_left = np.array((top_left_x, top_left_y))
    # top_right = np.array((top_right_x, top_right_y))
    # length = np.linalg.norm(top_left - top_right)
    return rectangle_properties(result)["length"]
