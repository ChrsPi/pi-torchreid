"""Tests for data loading and datasets."""

import pytest
import torch
from torchreid.data.datasets import Dataset, ImageDataset
from torchreid.data.sampler import build_train_sampler
from tests.fixtures.mock_datasets import create_mock_image_dataset


class TestDataset:
    """Test Dataset base class."""

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        train = [("path1.jpg", 0, 0, 0), ("path2.jpg", 1, 0, 0)]
        query = [("path3.jpg", 0, 1, 0)]
        gallery = [("path4.jpg", 0, 1, 0), ("path5.jpg", 1, 1, 0)]

        dataset = Dataset(train, query, gallery, mode="train")
        assert len(dataset) == 2
        assert dataset.num_train_pids == 2
        assert dataset.num_train_cams == 1

    def test_dataset_modes(self):
        """Test different dataset modes."""
        train = [("path1.jpg", 0, 0, 0), ("path2.jpg", 1, 0, 0)]
        query = [("path3.jpg", 0, 1, 0)]
        gallery = [("path4.jpg", 0, 1, 0)]

        # Test train mode
        dataset_train = Dataset(train, query, gallery, mode="train")
        assert len(dataset_train) == 2

        # Test query mode
        dataset_query = Dataset(train, query, gallery, mode="query")
        assert len(dataset_query) == 1

        # Test gallery mode
        dataset_gallery = Dataset(train, query, gallery, mode="gallery")
        assert len(dataset_gallery) == 1

    def test_dataset_combineall(self):
        """Test combineall functionality."""
        train = [("path1.jpg", 0, 0, 0)]
        query = [("path2.jpg", 0, 1, 0)]
        gallery = [("path3.jpg", 0, 1, 0)]

        dataset = Dataset(train, query, gallery, mode="train", combineall=True)
        # After combineall, train should include query and gallery
        assert len(dataset.train) > len(train)

    def test_dataset_add(self):
        """Test dataset concatenation."""
        train1 = [("path1.jpg", 0, 0, 0), ("path2.jpg", 1, 0, 0)]
        query1 = [("path3.jpg", 0, 1, 0)]
        gallery1 = [("path4.jpg", 0, 1, 0)]

        train2 = [("path5.jpg", 2, 0, 0), ("path6.jpg", 3, 0, 0)]
        query2 = [("path7.jpg", 2, 1, 0)]
        gallery2 = [("path8.jpg", 2, 1, 0)]

        dataset1 = Dataset(train1, query1, gallery1, mode="train")
        dataset2 = Dataset(train2, query2, gallery2, mode="train")

        combined = dataset1 + dataset2
        assert len(combined.train) == len(train1) + len(train2)

    def test_dataset_get_num_pids(self):
        """Test get_num_pids method."""
        train = [("path1.jpg", 0, 0, 0), ("path2.jpg", 1, 0, 0), ("path3.jpg", 0, 0, 0)]
        query = [("path4.jpg", 0, 1, 0)]
        gallery = [("path5.jpg", 0, 1, 0)]

        dataset = Dataset(train, query, gallery, mode="train")
        assert dataset.get_num_pids(train) == 2  # Two unique pids: 0 and 1

    def test_dataset_get_num_cams(self):
        """Test get_num_cams method."""
        train = [("path1.jpg", 0, 0, 0), ("path2.jpg", 1, 1, 0), ("path3.jpg", 0, 2, 0)]
        query = [("path4.jpg", 0, 1, 0)]
        gallery = [("path5.jpg", 0, 1, 0)]

        dataset = Dataset(train, query, gallery, mode="train")
        assert dataset.get_num_cams(train) == 3  # Three unique cams: 0, 1, 2


class TestImageDataset:
    """Test ImageDataset class."""

    def test_image_dataset_initialization(self, tmp_data_dir):
        """Test ImageDataset initialization with mock data."""
        train, query, gallery = create_mock_image_dataset(
            num_train=1,
            num_query=1,
            num_gallery=1,
            num_pids=1,
            num_cams=1,
            root_dir=tmp_data_dir,
        )
        dataset = ImageDataset(train, query, gallery, mode="train")
        assert len(dataset.train) == 1
        assert len(dataset.query) == 1
        assert len(dataset.gallery) == 1

    def test_image_dataset_inheritance(self):
        """Test that ImageDataset inherits from Dataset."""
        assert issubclass(ImageDataset, Dataset)


class TestSamplers:
    """Test data samplers."""

    def test_build_train_sampler_random(self):
        """Test building RandomSampler."""
        # Create mock data source
        data_source = [
            ("path1.jpg", 0, 0, 0),
            ("path2.jpg", 0, 0, 0),
            ("path3.jpg", 1, 0, 0),
            ("path4.jpg", 1, 0, 0),
        ]

        sampler = build_train_sampler(data_source, train_sampler="RandomSampler", batch_size=2)
        assert sampler is not None

    def test_build_train_sampler_random_identity(self):
        """Test building RandomIdentitySampler."""
        data_source = [
            ("path1.jpg", 0, 0, 0),
            ("path2.jpg", 0, 0, 0),
            ("path3.jpg", 1, 0, 0),
            ("path4.jpg", 1, 0, 0),
        ]

        sampler = build_train_sampler(
            data_source, train_sampler="RandomIdentitySampler", batch_size=4, num_instances=2
        )
        assert sampler is not None

    def test_build_train_sampler_invalid(self):
        """Test error handling for invalid sampler names."""
        data_source = [("path1.jpg", 0, 0, 0)]
        with pytest.raises(AssertionError, match="train_sampler must be one of"):
            build_train_sampler(data_source, train_sampler="InvalidSampler", batch_size=1)
