import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Literal
"""
this file provide dataloader for training and testing.
augmented images are stored in directory `tmp`

Raise error:
    invalid dataset
    
Returns:
    <dataloader>:CLRDataset
"""

class ViewGenerator:
    """Take some random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if self.n_views == 1:
            return self.base_transform(x)
        elif self.n_views > 1:
            return [self.base_transform(x) for _ in range(self.n_views)]


class CIFAR100(Dataset):
    def __init__(
        self,
        dataset_dir,
        transform=None,
        subset: Literal["train", "test"] = "train",
        device="cpu",
    ) -> None:
        path = os.path.join(dataset_dir, subset)
        path = path.replace("\\",'/')
        self.data_dict = self.unpickle(path)
        self.data = self.data_dict[b"data"]
        self.labels = self.data_dict[b"fine_labels"]
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        # resize test

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.from_numpy(img).view(3,32,32).permute(1,2,0))
        # plt.savefig('./tmp/cifar_raw.png')

        img = torch.from_numpy(img).to(device=self.device).view(3, 32, 32)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def unpickle(file, encoding="bytes"):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding=encoding)
        return dict


class ImageNet200(Dataset):
    def __init__(
        self,
        dataset_dir,
        transform=None,
        subset: Literal["train", "test", "val"] = "train",
        device="cpu",
    ) -> None:
        # load data without labels
        self.device = device
        self.subset = subset
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.data, self.labels = self.loadimages()

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.labels[index]
        # some image in the dataset is single channel(grayscale image)
        # e.g. data/tiny-imagenet-200/train/n04356056/images/n04356056_47.JPEG
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img).to(self.device)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    def loadimages(self, pattern="**/*.JPEG"):
        dataset_dir = Path(self.dataset_dir)
        with open(dataset_dir / "wnids.txt") as f:
            ids = f.read().split("\n")
        ids.remove("")
        label_dict = dict(zip(ids, range(200)))
        if self.subset == "train":
            img_paths = []
            img_labels = []
            for file in sorted((dataset_dir / "train").glob(pattern)):
                img_paths.append(file)
                img_labels.append(label_dict[file.stem.split("_")[0]])
            return img_paths, img_labels
        elif self.subset == "test":
            return list((dataset_dir / "test").glob(pattern)), []
        elif self.subset == "val":
            df = pd.read_csv(
                dataset_dir / "val/val_annotations.txt", delimiter="\t", header=None
            )
            img_paths = [
                dataset_dir / f"val/images/{file_name}" for file_name in df.iloc[:, 0]
            ]
            img_labels = [label_dict[class_name] for class_name in df.iloc[:, 1]]
            return img_paths, img_labels


class CLRDataset:

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # https://arxiv.org/abs/2002.05709

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                # transforms.RandomRotation(degrees=90),
            ]
        )
        return data_transforms

    @classmethod
    def get_dataset(self, name, n_views, device="cpu"):
        datasets_available = {
            "cifar100-train": CIFAR100(
                dataset_dir="./data/cifar-100-python",
                subset="train",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "cifar100-test": CIFAR100(
                dataset_dir="./data/cifar-100-python",
                subset="test",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-train": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="train",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-test": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="test",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
            "imagenet200-valid": ImageNet200(
                dataset_dir="./data/tiny-imagenet-200",
                subset="val",
                transform=ViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                device=device,
            ),
        }
        try:
            dataset = datasets_available[name]
        except KeyError:
            raise ValueError("Invalid dataset selection.")
        else:
            return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CLRDataset.get_dataset(name="cifar100-train", n_views=2)
    x, _ = dataset[0]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(x[1].permute(1, 2, 0).cpu().numpy())
    fig.savefig("./tmp/cifar_augmented.png")

    dataset = CLRDataset.get_dataset(
        name="imagenet200-valid", n_views=2,
    )
    x, _ = dataset[0]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(x[1].permute(1, 2, 0).cpu().numpy())
    fig.savefig("./tmp/imagenet_augmented.png")
