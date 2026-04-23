from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_pipeline.labels import load_image_mask_pairs


class CrowdDataset(Dataset):
    def __init__(self):
        # load all image-mask pairs from labels.py
        self.pairs = load_image_mask_pairs()

        # convert dictionary into a list so we can index it with numbers
        self.names = list(self.pairs.keys())

    def __len__(self):
        # total number of samples in the dataset
        return len(self.names)

    def __getitem__(self, index):
        # get one filename key, like "I_1"
        name = self.names[index]

        # get its image path and mask path
        paths = self.pairs[name]
        image_path = paths["image"]
        mask_path = paths["mask"]

        # load image in color
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # load mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # check that both files loaded correctly
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# resize both image and mask to the same fixed size
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # convert image from numpy shape (H, W, C) to torch shape (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # convert mask to 0 and 1
        mask = (mask > 0).astype(np.float32)

        # add channel dimension so mask becomes (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    dataset = CrowdDataset()

    print("Number of samples:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", torch.unique(mask))