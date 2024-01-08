# Module to pre-process images before sent into training
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
# Custom modules
import gen_class
import image_editor

ImageUtil = image_editor.ImageUtil

# Art data set to hold images being used by the model
class ArtDataset(Dataset):
    def __init__(self, data_frame, root_dir, augmentFlag=False):
        """
        Args:
            data_frame (pandas.DataFrame): DataFrame containing file paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.augmentFlag = augmentFlag

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        try:
            # Construct the full path to the image
            relative_img_path = self.data_frame.iloc[idx, 0]
            full_img_path = os.path.join(self.root_dir, relative_img_path)
            class_id = self.data_frame.iloc[idx, 1]
            class_name = gen_class.get_class_name(class_id)

            # Custom augments
            opened_image = ImageUtil.open(full_img_path)
            resized_image = ImageUtil.resize(opened_image, 512)
            normalized_image = ImageUtil.to_tensor_normalize(resized_image)

            ## Training sets augment further
            if self.augmentFlag == True:
                augmented_image = ImageUtil.augment(normalized_image)
                return augmented_image, class_id
            else:
                return normalized_image, class_id
        except (FileNotFoundError, OSError):
            return None, None

