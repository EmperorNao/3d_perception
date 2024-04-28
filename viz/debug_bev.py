from dataset.cadc import CadcDataset
from bev import draw_bev
import numpy as np


dataset_path = "/home/emperornao/p/perception/data/cadc"
scene_sample = None
for scene in CadcDataset.lazy_create_from_path(dataset_path):
    scene_sample = scene
    break

image = draw_bev(scene_sample.lidar_cloud,
         [],
         [1000, 1000],
         [50, 50])
