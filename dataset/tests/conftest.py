import os
import json
import pytest


@pytest.fixture
def dataset_path():
    return os.path.join(os.environ['PROJECT_ROOT_DIR'], "resources/test_data/cadc")


@pytest.fixture
def empty_dataset_path():
    return os.path.join(os.environ['PROJECT_ROOT_DIR'], "resources/test_data/empty_cadc")


@pytest.fixture
def cuboid():
    cuboid_path = os.path.join(os.environ['PROJECT_ROOT_DIR'], "resources/test_data/cadc_structures/bboxes.json")
    with open(cuboid_path) as fr:
        return json.load(fr)[0]['cuboids'][0]


@pytest.fixture
def pcl_path():
    return os.path.join(os.environ['PROJECT_ROOT_DIR'], "resources/test_data/cadc_structures/pc.bin")


@pytest.fixture
def intrinsics_path():
    return os.path.join(os.environ['PROJECT_ROOT_DIR'], "resources/test_data/cadc_structures/intr.yaml")
