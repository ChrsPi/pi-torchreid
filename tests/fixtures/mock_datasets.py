"""Mock dataset utilities for testing without real data."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def create_mock_image(width=128, height=256, channels=3):
    """Create a mock PIL Image for testing."""
    if channels == 3:
        mode = "RGB"
    elif channels == 1:
        mode = "L"
    else:
        raise ValueError(f"Unsupported channels: {channels}")

    # Create a random image
    img_array = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    if channels == 1:
        img_array = img_array.squeeze(-1)
    return Image.fromarray(img_array, mode=mode)


def create_mock_image_dataset(
    num_train=100,
    num_query=20,
    num_gallery=80,
    num_pids=10,
    num_cams=3,
    root_dir=None,
):
    """
    Generate synthetic image dataset tuples.

    Returns:
        tuple: (train, query, gallery) where each is a list of tuples:
            (img_path, pid, camid) or (img_path, pid, camid, dsetid)
    """
    if root_dir is None:
        root_dir = tempfile.mkdtemp()

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    train = []
    query = []
    gallery = []

    # Create train set
    for i in range(num_train):
        pid = i % num_pids
        camid = i % num_cams
        img_path = root_dir / f"train_{i:04d}_pid{pid}_cam{camid}.jpg"
        img = create_mock_image()
        img.save(img_path)
        train.append((str(img_path), pid, camid, 0))

    # Create query set (subset of identities)
    query_pids = list(range(min(num_pids, num_query)))
    for i, pid in enumerate(query_pids):
        camid = i % num_cams
        img_path = root_dir / f"query_{i:04d}_pid{pid}_cam{camid}.jpg"
        img = create_mock_image()
        img.save(img_path)
        query.append((str(img_path), pid, camid, 0))

    # Create gallery set
    for i in range(num_gallery):
        pid = i % num_pids
        camid = i % num_cams
        img_path = root_dir / f"gallery_{i:04d}_pid{pid}_cam{camid}.jpg"
        img = create_mock_image()
        img.save(img_path)
        gallery.append((str(img_path), pid, camid, 0))

    return train, query, gallery


def create_mock_video_dataset(
    num_train=50,
    num_query=10,
    num_gallery=40,
    num_pids=10,
    num_cams=3,
    seq_len=15,
    root_dir=None,
):
    """
    Generate synthetic video dataset tuples.

    Returns:
        tuple: (train, query, gallery) where each is a list of tuples:
            (video_path, pid, camid) or (video_path, pid, camid, dsetid)
            where video_path is a directory containing frame images
    """
    if root_dir is None:
        root_dir = tempfile.mkdtemp()

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    train = []
    query = []
    gallery = []

    # Create train set
    for i in range(num_train):
        pid = i % num_pids
        camid = i % num_cams
        video_dir = root_dir / f"train_{i:04d}_pid{pid}_cam{camid}"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Create sequence of frames
        for frame_idx in range(seq_len):
            frame_path = video_dir / f"frame_{frame_idx:04d}.jpg"
            img = create_mock_image()
            img.save(frame_path)

        train.append((str(video_dir), pid, camid, 0))

    # Create query set
    query_pids = list(range(min(num_pids, num_query)))
    for i, pid in enumerate(query_pids):
        camid = i % num_cams
        video_dir = root_dir / f"query_{i:04d}_pid{pid}_cam{camid}"
        video_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(seq_len):
            frame_path = video_dir / f"frame_{frame_idx:04d}.jpg"
            img = create_mock_image()
            img.save(frame_path)

        query.append((str(video_dir), pid, camid, 0))

    # Create gallery set
    for i in range(num_gallery):
        pid = i % num_pids
        camid = i % num_cams
        video_dir = root_dir / f"gallery_{i:04d}_pid{pid}_cam{camid}"
        video_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(seq_len):
            frame_path = video_dir / f"frame_{frame_idx:04d}.jpg"
            img = create_mock_image()
            img.save(frame_path)

        gallery.append((str(video_dir), pid, camid, 0))

    return train, query, gallery
