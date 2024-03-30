import yaml
from yaml.loader import SafeLoader
import wget
import zipfile
import logging
import argparse
from os import remove
from typing import Dict
from os.path import join
from pathlib import Path


def unzip(file_path: str, remove_: bool = False):
    archive = zipfile.ZipFile(file_path)
    archive.extractall(parent(file_path))
    archive.close()
    if remove_:
        remove(file_path)


def mkdir(directory: str, exist_ok: bool = False):
    Path(directory).mkdir(parents=True, exist_ok=exist_ok)


def parent(path: str) -> Path:
    return Path(path).parent.absolute()


BASE_LINK = 'BASE_LINK'
DATA = 'DATA'

ANN_FILE = '3d_ann.json'
CALIB_FILE = 'calib.zip'
LABEL_FILE = 'labeled.zip'


def download(base_dir: str, data: Dict):

    mkdir(base_dir, exist_ok=True)
    base_link = data[BASE_LINK]
    cadc_dict = data[DATA]
    base_path = join(base_dir, "cadc")
    mkdir(base_path, True)

    logging.info("Downloading CADC dataset in directory %s" % base_path)
    for date in cadc_dict:

        date_path = join(base_path, date)
        mkdir(date_path, True)
        logging.info("Date %s" % date)

        calib_url = join(base_link, date, CALIB_FILE)
        calib_filename = wget.download(calib_url, join(date_path, CALIB_FILE))
        unzip(calib_filename, True)

        for drive in cadc_dict[date]:

            drive_path = join(date_path, drive)
            mkdir(drive_path, True)
            logging.info("Drive %s" % drive)

            ann_3d_url = join(base_link, date, drive, ANN_FILE)
            wget.download(ann_3d_url, join(drive_path, ANN_FILE))

            data_url = join(base_link, date, drive, LABEL_FILE)
            data_filename = wget.download(data_url, join(drive_path, LABEL_FILE))

            unzip(data_filename, True)


def main():
    parser = argparse.ArgumentParser(description='download')
    parser.add_argument("--dataset_dict", default="dataset/configs/format.yaml", type=str)
    parser.add_argument('--base_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    with open(args.dataset_dict) as fr:
        data = yaml.load(fr.read(), Loader=SafeLoader)
    download(args.base_dir, data)


if __name__ == "__main__":
    main()
