import os
import yaml
import json

import numpy as np
from PIL import Image as PILImage

from typing import List, Callable
from dataclasses import dataclass

from .structures import (
    Scene, SceneID, RideID, Camera, Image,
    LidarCloud, BBox, Transform, GT3D
)


@dataclass
class CadcBBox(BBox):
    @staticmethod
    def from_cuboid(cuboid):
        return BBox(
            cuboid['label'],
            np.array([cuboid['position']['x'],
                      cuboid['position']['y'],
                      cuboid['position']['z']]
                     ),
            np.array([cuboid['dimensions']['x'],
                      cuboid['dimensions']['y'],
                      cuboid['dimensions']['z']]
                     ),
            cuboid['yaw'],
            cuboid['stationary']
        )


@dataclass
class CadcLidarCloud(LidarCloud):

    @property
    def intensity(self):
        return self.points[:, 3]

    @staticmethod
    def from_binary(file_name: str):
        return CadcLidarCloud(np.fromfile(file_name, dtype=np.float32).reshape(-1, 4))


@dataclass
class CadcIntrinsics(Transform):
    @staticmethod
    def from_yaml(path):
        with open(path) as fr:
            data = yaml.load(fr, yaml.SafeLoader)
        camera_name = data['camera_name']
        return Transform(
            camera_name + "_intrinsics",
            np.array(data['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
        )


@dataclass
class CadcDataset:
    NUMBER_OF_CAMERAS = 8
    scenes: List[Scene]

    @staticmethod
    def create_from_path(path, grep: Callable = None):

        scenes = [scene for scene in CadcDataset.lazy_create_from_path(path, grep)]
        return scenes

    @staticmethod
    def lazy_create_from_path(path, grep: Callable = None):

        dates = os.listdir(path)
        for date in dates:
            date_path = os.path.join(path, date)

            calib_path = os.path.join(date_path, "calib")
            cam2calib = {}
            for camera_idx in range(CadcDataset.NUMBER_OF_CAMERAS):
                cam2calib[camera_idx] = CadcIntrinsics.from_yaml(
                    os.path.join(calib_path, str(camera_idx).rjust(2, '0') + ".yaml")
                )

            # TODO: read extrinsics # noqa

            rides = list(filter(lambda dir_name: dir_name != "calib",
                                os.listdir(date_path)))
            for ride in rides:
                ride_path = os.path.join(date_path, ride)
                ride_id = RideID(date, ride)

                cuboids_path = os.path.join(ride_path, "3d_ann.json")
                with open(cuboids_path) as fr:
                    cuboids = json.load(fr)

                lidar_path = os.path.join(ride_path, "labeled", "lidar_points")
                image_base_path = os.path.join(ride_path, "labeled", "image_{}")

                scene_ids = list(map(
                    lambda file_name: file_name.rstrip(".bin"),
                    os.listdir(os.path.join(lidar_path, "data"))
                ))

                for scene_idx, scene_id in enumerate(scene_ids):
                    if grep and not grep(SceneID(ride_id, scene_id)):
                        continue
                    cameras = []
                    for camera_idx in range(CadcDataset.NUMBER_OF_CAMERAS):
                        image_path = os.path.join(
                            image_base_path.format(str(camera_idx).rjust(2, "0")),
                            "data",
                            scene_id + ".png"
                        )
                        calib = cam2calib[camera_idx]
                        cam_name = calib.name.rstrip("_intrinsics")
                        cameras.append(Camera(
                            cam_name,
                            calib,
                            [],
                            Image(PILImage.open(image_path))
                        ))

                    cubs = cuboids[scene_idx]['cuboids']
                    bboxes = []
                    for cuboid in cubs:
                        bboxes.append(CadcBBox.from_cuboid(cuboid))

                    yield Scene(
                        SceneID(ride_id, scene_id),
                        cameras,
                        CadcLidarCloud.from_binary(
                            os.path.join(lidar_path, "data", scene_id + ".bin")
                        ),
                        GT3D(bboxes)
                    )

    @staticmethod
    def get_rides_info_list(path) -> List[RideID]:
        rides_info_list = []
        dates = os.listdir(path)
        for date in dates:
            date_path = os.path.join(path, date)
            rides = list(filter(lambda dir_name: dir_name != "calib",
                                os.listdir(date_path)))
            for ride in rides:
                ride_id = RideID(date, ride)
                rides_info_list.append(ride_id)
        return rides_info_list

    @staticmethod
    def get_scenes_info_list(path) -> List[SceneID]:
        scenes_info_list = []
        dates = os.listdir(path)
        for date in dates:
            date_path = os.path.join(path, date)
            rides = list(filter(lambda dir_name: dir_name != "calib",
                                os.listdir(date_path)))
            for ride in rides:
                ride_path = os.path.join(date_path, ride)
                ride_id = RideID(date, ride)
                lidar_path = os.path.join(ride_path, "labeled", "lidar_points")
                scene_ids = list(map(
                    lambda file_name: file_name.rstrip(".bin"),
                    os.listdir(os.path.join(lidar_path, "data"))
                ))
                for scene_id in scene_ids:
                    scenes_info_list.append(SceneID(ride_id, scene_id))
        return scenes_info_list
