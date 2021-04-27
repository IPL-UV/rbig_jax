import pytorch_lightning as pl
from torchvision.datasets import CIFAR
from rbig_jax.custom_types import ImageShape
from rbig_jax.transforms.reshape import flatten_image, unflatten_image
from typing import Optional
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from rbig_jax.data import GenericDataset
from pyprojroot import here

# spyder up to find the root
from pathlib import Path

root = here(project_files=[".here"])


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        val_split: float = 0.2,
        seed: int = 123,
        flatten: bool = True,
        subset: Optional[int] = None,
        dataset_dir: str = None,
    ):

        self.val_split = val_split
        self.batch_size = batch_size
        self.seed = seed
        self.flatten = flatten
        self.image_shape = ImageShape(C=3, H=32, W=32)
        self.subset = subset
        if dataset_dir is None:
            dataset_dir = str(Path(root).joinpath("datasets/cifar"))
        self.dataset_dir = dataset_dir

        self.filters = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]

    def prepare_data(self):
        # download
        self.train_dataset = CIFAR(self.dataset_dir, download=True, train=True)

        # self.test_dataset = MNIST(os.getcwd(), download=True, train=False)

    def setup(self, stage=None):

        # assign train/val split
        Xtrain, Xval = train_test_split(
            self.train_dataset.data, test_size=self.val_split, random_state=self.seed,
        )

        Xtrain = Xtrain.numpy().astype(np.float32)
        Xval = Xval.numpy().astype(np.float32)

        if self.subset is not None:
            Xtrain = Xtrain[: self.subset]

        if self.flatten:
            Xtrain = flatten_image(Xtrain, self.image_shape, batch=True)
            Xval = flatten_image(Xval, self.image_shape, batch=True)

        self.Xtrain = Xtrain
        self.Xval = Xval
        self.ds_train = GenericDataset(Xtrain)
        self.ds_val = GenericDataset(Xval)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def valid_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
