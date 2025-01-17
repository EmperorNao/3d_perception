import numpy as np


def rotate_points_2d(points: np.array, angle: np.array) -> np.array:
    """
    :param points: array of shape (b, n, 2)
        b - batch, n - points to translate, 2 - dimensions (x, y)
    :param angle: array of size b with angles
    :return: return array of shape (b, n, 2) in which all points in batch rotated by angle
    """
    rot = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])

    return points @ rot.transpose(2, 0, 1)


def rotate_points_3d(points: np.array, angle: np.array, axis: str) -> np.array:
    """
    :param points: array of shape (b, n, 3)
        b - batch, n - points to translate, 3 - dimensions (x, y, z)
    :param angle: array of size b with angles
    :param axis: string one of 'x', 'y' or 'z' in which dim to rotate points
    :return: return array of shape (b, n, z) in which all points in batch rotated by angle
    """
    assert axis in ['x', 'y', 'z']
    if axis == 'z':
        rot = np.array([
            [np.cos(angle), -np.sin(angle), np.zeros_like(angle)],
            [np.sin(angle), np.cos(angle), np.zeros_like(angle)],
            [np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)]
        ])
    if axis == 'y':
        rot = np.array([
            [np.cos(angle), np.zeros_like(angle), np.sin(angle)],
            [np.zeros_like(angle), np.ones_like(angle), np.zeros_like(angle)],
            [-np.sin(angle), np.zeros_like(angle), np.cos(angle)],
        ])
    if axis == 'x':
        rot = np.array([
            [np.ones_like(angle), np.zeros_like(angle), np.zeros_like(angle)],
            [np.zeros_like(angle), np.cos(angle), -np.sin(angle)],
            [np.zeros_like(angle), np.sin(angle), np.cos(angle)],
        ])

    return points @ rot.transpose(2, 0, 1)


def boxes_straight2rotated(boxes: np.array) -> np.array:
    """Rotate 2b boxes
        :param boxes: np.array with shape (n,5) in format [cx, cy, dx, dy, yaw]
        :return np.array with 4 rotated points for each box, shape (n,4,2)
    """

    cx, cy, dx, dy, yaw = boxes[:, :5].transpose(1, 0)
    dx, dy = dx / 2, dy / 2
    new_d = translate_points_2d(np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]).transpose(2, 0, 1), yaw)
    rotated_boxes = new_d + np.expand_dims(np.array([cx, cy]), 0).transpose(2, 0, 1)

    return rotated_boxes


def boxes_straight2rotated_3d(boxes: np.array) -> np.array:
    """Rotate 3d boxes in z dim
        :param boxes: np.array with shape (n,7) in format [cx, cy, cz, dx, dy, dz, yaw]
        :return np.array with 8 rotated points for each box, shape (n,8,3)
    """

    cx, cy, cz, dx, dy, dz, yaw = boxes[:, :7].transpose(1, 0)
    dx, dy, dz = dx / 2, dy / 2, dz / 2
    new_d = rotate_points_3d(np.array([
        [-dx, -dy, -dz],
        [-dx, dy, -dz],
        [-dx, dy, dz],
        [-dx, -dy, dz],
        [dx, -dy, -dz],
        [dx, dy, -dz],
        [dx, dy, dz],
        [dx, -dy, dz]
    ]).transpose(2, 0, 1), yaw, axis='z')

    rotated_boxes = new_d + np.expand_dims(np.array([cx, cy, cz]), 0).transpose(2, 0, 1)

    return rotated_boxes


def boxes_rotated2cxcydxdy(
        boxes: np.array
):
    """
    :param boxes: array of 2d rotated bounding boxes with 4 points with shape (b, 4, 2)
    :return: array of 2d bounding boxes in format (cx, cy, dx, dy) with shape (b, 4)
    """

    x1 = np.expand_dims(np.min(boxes[:, :, 0], axis=1), -1)
    x2 = np.expand_dims(np.max(boxes[:, :, 0], axis=1), -1)
    y1 = np.expand_dims(np.min(boxes[:, :, 1], axis=1), -1)
    y2 = np.expand_dims(np.max(boxes[:, :, 1], axis=1), -1)

    wdx = np.abs(x2 - x1) / 2
    wdy = np.abs(y2 - y1) / 2
    cx = x1 + wdx
    cy = y1 + wdy

    dx = np.expand_dims(np.linalg.norm(boxes[:, 0] - boxes[:, 1], ord=2, axis=1), -1)
    dy = np.expand_dims(np.linalg.norm(boxes[:, 1] - boxes[:, 2], ord=2, axis=1), -1)

    return np.concatenate(
        (cx, cy, dx, dy),
        axis=1
    )
