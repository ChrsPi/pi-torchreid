"""VeRi-CARLA synthetic vehicle re-identification dataset.

Reference:
    Kaneko et al. VeRi-CARLA: A Large-Scale Simulated Dataset for
    Vehicle Re-Identification in CARLA. 2023.

URL: https://github.com/sekilab/VehicleReIdentificationDataset

Dataset statistics:
    - identities: 600 (train) + 50 (query/gallery).
    - images: 50949 (train) + 424 (query) + 3823 (gallery).

Filename format: {timestamp}_{frameindex}_{vehicleID}.jpg
    - timestamp (14 digits): date+time of capture.
    - frameindex: sequential frame number within a sequence.
    - vehicleID: vehicle identity label.

Camera IDs: Derived from (date, frame_index % 2) to create 12 pseudo-cameras.
This ensures query and gallery images of the same vehicle are assigned to
different cameras, which is required by standard re-ID evaluation metrics.
"""

import glob
import os.path as osp
import re

from pi_torchreid.utils import logger

from ..dataset import ImageDataset


class VeRiCARLA(ImageDataset):
    """VeRi-CARLA synthetic vehicle re-ID dataset."""

    dataset_dir = ""  # data sits directly under root

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = self.root

        self.train_dir = osp.join(self.data_dir, "image_train")
        self.query_dir = osp.join(self.data_dir, "image_query")
        self.gallery_dir = osp.join(self.data_dir, "image_gallery")

        required_files = [self.train_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        # Build date→base_camid mapping from all splits
        all_dates: set[str] = set()
        for d in [self.train_dir, self.query_dir, self.gallery_dir]:
            for fname in glob.glob(osp.join(d, "*.jpg")):
                date = osp.basename(fname)[:8]  # YYYYMMDD
                all_dates.add(date)
        self._date2basecam = {date: i * 2 for i, date in enumerate(sorted(all_dates))}

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        super().__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"(\d{14})_(\d+)_(\d+)\.jpg")

        pid_container = set()
        for img_path in img_paths:
            basename = osp.basename(img_path)
            match = pattern.search(basename)
            if match is None:
                logger.warning(
                    "Skipping VeRi-CARLA file '%s' in '%s': expected pattern '%s'",
                    basename,
                    dir_path,
                    pattern.pattern,
                )
                continue
            pid = int(match.group(3))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            basename = osp.basename(img_path)
            match = pattern.search(basename)
            if match is None:
                logger.warning(
                    "Skipping VeRi-CARLA file '%s' in '%s': expected pattern '%s'",
                    basename,
                    dir_path,
                    pattern.pattern,
                )
                continue
            timestamp = match.group(1)
            frame_idx = int(match.group(2))
            pid = int(match.group(3))
            date = timestamp[:8]
            # Pseudo-camera: base_cam (from date) + frame parity (0 or 1)
            camid = self._date2basecam[date] + (frame_idx % 2)
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
