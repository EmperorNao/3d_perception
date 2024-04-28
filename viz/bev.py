import math
import numpy as np
from PIL import Image
from typing import List, Tuple

from dataset import LidarCloud, BBox
from geometry.transformations import rotate_points_2d


def create_empty_bev(
        image_size: List[int],
        color: List[int] = None
):
    if color is None:
        color = [0, 0, 0]
    assert len(color) == 3
    image = np.zeros(list(image_size) + [3], dtype=np.uint8)
    image[:, :] = np.array(color)
    return image


def draw_pcl_on_bev(
        image: np.array,
        pcl: LidarCloud,
        scene_size: List[int],
        points_colors: np.array = None,
) -> np.array:

    points = pcl.xyz
    image_size = list(image.shape[:2])
    if points_colors is None:
        points_colors = np.zeros([points.shape[0], 3], dtype=np.int8)
        points_colors.fill(255)

    active_points_indexes = np.logical_and(
        np.abs(points[:, 0]) < scene_size[0],
        np.abs(points[:, 1]) < scene_size[1],
    )
    filtered_points = points[active_points_indexes]
    points_colors = points_colors[active_points_indexes]

    w_step = 2 * scene_size[0] / image_size[0]
    h_step = 2 * scene_size[1] / image_size[1]
    x = np.floor((filtered_points[:, 0] + scene_size[0]) / w_step).astype(np.int32)
    y = np.floor((filtered_points[:, 1  ] + scene_size[1]) / h_step).astype(np.int32)

    coords = np.vstack([x, y]).T

    for (x, y), color in zip(coords, points_colors):
        image[y, x, :] = color

    return image


def draw_bboxes_on_bev(
        image: np.array,
        bboxes: List[BBox],
        scene_size: List[int],
        boxes_colors: np.array = None,
) -> np.array:

    image_size = list(image.shape[:2])

    angles = []
    points = []
    centers = []
    for bbox in bboxes:
        angles.append(bbox.yaw)
        points.append(bbox.vertices_2d - bbox.dimensions[:2])
        centers.append(bbox.dimensions[:2])

    rotated_points = rotate_points_2d(np.array(points), angles)

    active_points_indexes = np.logical_and(
        np.abs(points[:, 0]) < scene_size[0],
        np.abs(points[:, 1]) < scene_size[1],
    )
    filtered_points = points[active_points_indexes]
    points_colors = points_colors[active_points_indexes]

    w_step = 2 * scene_size[0] / image_size[0]
    h_step = 2 * scene_size[1] / image_size[1]
    x = np.floor((filtered_points[:, 0] + scene_size[0]) / w_step).astype(np.int32)
    y = np.floor((filtered_points[:, 1] + scene_size[1]) / h_step).astype(np.int32)

    coords = np.vstack([x, y]).T

    for (x, y), color in zip(coords, points_colors):
        image[y, x, :] = color

    return image


def save_image(image, save_path):
    pil_image = Image.fromarray(image).convert('RGB')
    pil_image.save(save_path)
