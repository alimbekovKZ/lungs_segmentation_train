import numpy as np
import cv2

from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, names, data_folder, mask_folder, mode="train", transforms=None):
        self.names = names
        self.transforms = transforms
        self.mode = mode
        self.data_folder = data_folder
        self.mask_folder = mask_folder

    def __getitem__(self, idx):
        if self.names[idx].endswith("_mask.png"):
            image = cv2.imread(
                "{}/{}".format(
                    self.data_folder, self.names[idx].replace("_mask.png", ".png")
                )
            )
        else:
            image = cv2.imread("{}/{}".format(self.data_folder, self.names[idx]))
        image = (image - image.min()) / (image.max() - image.min())
        left = cv2.imread(
            "{}/ml/{}".format(self.mask_folder, self.names[idx]), cv2.IMREAD_GRAYSCALE
        )
        right = cv2.imread(
            "{}/mr/{}".format(self.mask_folder, self.names[idx]), cv2.IMREAD_GRAYSCALE
        )
        mask = np.concatenate(
            (np.expand_dims(left, axis=2), np.expand_dims(right, axis=2)), axis=2
        )
        if mask.max() == 255:
            mask = (mask / 255).astype("uint8")
        if self.transforms:
            augs = self.transforms(image=image, mask=mask)
        return augs["image"], augs["mask"]  # , self.names[idx][:-9]

    def __len__(self):
        return len(self.names)
