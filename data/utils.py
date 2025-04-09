from PIL import Image

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