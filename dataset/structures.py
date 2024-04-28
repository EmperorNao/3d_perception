import numpy as np
from PIL import Image as PILImage

from typing import List
from dataclasses import dataclass


@dataclass
class BBox:
    label: str
    position: np.array
    dimensions: np.array
    yaw: float
    stationary: bool

    @property
    def cx(self):
        return self.position[0]

    @property
    def cy(self):
        return self.position[1]

    @property
    def cz(self):
        return self.position[2]

    @property
    def dx(self):
        return self.dimensions[0] / 2

    @property
    def dy(self):
        return self.dimensions[1] / 2

    @property
    def dz(self):
        return self.dimensions[2] / 2

    @property
    def vertices_2d(self):
        return np.array([
            [self.cx - self.dx, self.cy - self.dy],
            [self.cx - self.dx, self.cy + self.dy],
            [self.cx + self.dx, self.cy + self.dy],
            [self.cx + self.dx, self.cy - self.dy],
        ])


@dataclass
class GT3D:
    boxes: List[BBox]


@dataclass
class Image:
    data: PILImage

    @property
    def width(self):
        return self.data.size[0]

    @property
    def height(self):
        return self.data.size[1]


@dataclass
class Transform:
    name: str
    data: np.array


@dataclass
class Camera:
    camera_name: str
    intrinsics: Transform
    transformations: List[Transform]
    image: Image


@dataclass
class LidarCloud:
    points: np.array

    @property
    def xyz(self):
        return self.points[:, :3]


@dataclass
class RideID:
    date: str
    ride_id: str

    def __str__(self):
        return self.date + "_" + self.ride_id


@dataclass
class SceneID:
    ride_id: RideID
    scene_id: str

    def __str__(self):
        return str(self.ride_id) + "_" + str(self.scene_id)


@dataclass
class Scene:
    scene_id: SceneID
    cameras: List[Camera]
    lidar_cloud: LidarCloud
    gt3d: GT3D
