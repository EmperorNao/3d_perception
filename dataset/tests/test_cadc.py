import numpy as np
from dataset.cadc import CadcDataset, CadcBBox, CadcLidarCloud, CadcIntrinsics


def test_bbox_load(cuboid):
    bbox = CadcBBox.from_cuboid(cuboid)
    assert bbox.label == "Truck"
    assert np.all(bbox.position == np.array([
        -19.10966344067156,
        -0.9994196630418124,
        -0.28325530444985225]))
    assert np.all(bbox.dimensions == np.array([
        2.609,
        21.222,
        4.31]))
    assert bbox.yaw == 0.2475953980193629
    assert bbox.stationary is False


def test_pcl_load(pcl_path):
    pcl = CadcLidarCloud.from_binary(pcl_path)
    assert pcl.points.shape == (2, 4)
    assert pcl.xyz.shape == (2, 3)
    assert pcl.intensity.shape == (2, )
    assert np.all(pcl.points == np.array([
        [1.0, 1.0, 1.0, 3.0],
        [0.0, -1.5, 2.3, 0.25323]],
        dtype=np.float32))


def test_intrinsics_load(intrinsics_path):
    params = CadcIntrinsics.from_yaml(intrinsics_path)
    assert params.name == "camera_F_intrinsics"
    assert params.data.shape == (3, 3)
    assert np.all(params.data == np.array(
        [[653.956033188809, -0.235925653043616, 653.221172545916],
         [0, 655.540088617960, 508.732863993917],
         [0, 0, 1]],
        dtype=np.float32
    ))


def test_cadc_empty_scenes_list(empty_dataset_path):
    assert len(CadcDataset.get_scenes_info_list(empty_dataset_path)) == 0


def test_cadc_empty_rides_list(empty_dataset_path):
    assert len(CadcDataset.get_rides_info_list(empty_dataset_path)) == 0


def test_cadc_scenes_list(dataset_path):
    assert len(CadcDataset.get_scenes_info_list(dataset_path)) == 5


def test_cadc_rides_list(dataset_path):
    assert len(CadcDataset.get_rides_info_list(dataset_path)) == 4


def test_cadc_scenes(dataset_path):
    def grep_func(scene_id):
        ride_id = scene_id.ride_id
        if ride_id.date == "2018_03_06" and ride_id.ride_id == "0001" and \
                scene_id.scene_id == "0000000000":
            return True
        return False
    scenes = CadcDataset.create_from_path(dataset_path, grep_func)
    assert len(scenes) == 1

    scene = scenes[0]
    assert str(scene.scene_id) == "2018_03_06_0001_0000000000"
    assert len(scene.cameras) == 8

    image = scene.cameras[0].image
    assert image.width != 0 and image.height != 0

    pcl = scene.lidar_cloud
    assert pcl.points.shape[0] != 0 and pcl.points.shape[1] == 4

    gt = scene.gt3d
    assert len(gt.boxes) != 0
