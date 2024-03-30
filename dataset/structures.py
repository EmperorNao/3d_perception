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
