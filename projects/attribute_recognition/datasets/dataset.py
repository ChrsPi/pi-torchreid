import os.path as osp

from pi_torchreid.utils import read_image


class Dataset:
    def __init__(self, train, val, test, attr_dict, transform=None, mode="train", verbose=True, **kwargs):
        self.train = train
        self.val = val
        self.test = test
        self._attr_dict = attr_dict
        self._num_attrs = len(self.attr_dict)
        self.transform = transform

        if mode == "train":
            self.data = self.train
        elif mode == "val":
            self.data = self.val
        else:
            self.data = self.test

        if verbose:
            self.show_summary()

    @property
    def num_attrs(self):
        return self._num_attrs

    @property
    def attr_dict(self):
        return self._attr_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, attrs = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, attrs, img_path

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError(f'"{fpath}" is not found')

    def show_summary(self):
        num_train = len(self.train)
        num_val = len(self.val)
        num_test = len(self.test)
        num_total = num_train + num_val + num_test

        print(f"=> Loaded {self.__class__.__name__}")
        print("  ------------------------------")
        print("  subset   | # images")
        print("  ------------------------------")
        print(f"  train    | {num_train:8d}")
        print(f"  val      | {num_val:8d}")
        print(f"  test     | {num_test:8d}")
        print("  ------------------------------")
        print(f"  total    | {num_total:8d}")
        print("  ------------------------------")
        print(f"  # attributes: {len(self.attr_dict)}")
        print("  attributes:")
        for label, attr in self.attr_dict.items():
            print(f"    {label:3d}: {attr}")
        print("  ------------------------------")
