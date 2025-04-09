from PIL import Image
import torch

class LeftCrop:
    def __init__(self, size):
        self.size = size  

    def __call__(self, img: Image.Image):
        width, height = img.size
        crop_width, crop_height = self.size

        # Ensure image is large enough
        if width < crop_width or height < crop_height:
            raise ValueError(f"Image size {img.size} is smaller than crop size {self.size}")

        # Calculate left crop coordinates
        left = 0
        upper = (height - crop_height) // 2
        right = crop_width
        lower = upper + crop_height

        return img.crop((left, upper, right, lower))
    
def pad_or_truncate(tensor, length=24):
    T, C, H, W = tensor.shape
    if T == length:
        return tensor
    elif T > length:
        return tensor[:length]
    else:
        pad_size = length - T
        padding = torch.zeros(pad_size, C, H, W) 
        return torch.cat((tensor, padding), dim=0)