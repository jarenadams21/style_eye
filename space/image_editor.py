import random
import torch
from torchvision import transforms
from PIL import Image

"""
    Uses torchvision to augment images before using in the model
"""
class ImageUtil():
    # Load an image file. Return the image as a tensor
    @staticmethod
    def open(image_file):
        image = Image.open(image_file)
        return image

    # Resize the image and maintain the aspect ratio
    @staticmethod
    def resize(image, target_size=512):
        # Calculate the new size, maintaining aspect ratio
        old_width, old_height = image.size
        aspect_ratio = old_width / old_height
        if old_width < old_height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        image = transforms.Resize((new_height, new_width))(image)

        # Calculate padding to make the image square
        pad_horizontal = (target_size - new_width) // 2
        pad_vertical = (target_size - new_height) // 2
        padding = (pad_horizontal, pad_vertical, target_size - new_width - pad_horizontal, target_size - new_height - pad_vertical)

        # Pad the image
        image = transforms.Pad(padding, fill=0, padding_mode='constant')(image)

        return image

    # Convert the image to a tensor and normalize it
    @staticmethod
    def to_tensor_normalize(image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    # Apply random transformations for data augmentation
    @staticmethod
    def augment(image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ])
        return transform(image)

# Example usage:
# image = ImageUtil.open('path/to/image.jpg')
# image = ImageUtil.resize(image, size=(256, 256))
# image = ImageUtil.to_tensor_normalize(image)
# image = ImageUtil.augment(image)
