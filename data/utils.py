from PIL import Image

class LeftCrop:
    def __init__(self, size):
        self.size = size  # assumes size is an int (for square crop)

    def __call__(self, img: Image.Image):
        width, height = img.size
        crop_size = self.size

        # Ensure image is large enough
        if width < crop_size or height < crop_size:
            raise ValueError(f"Image size {img.size} is smaller than crop size {crop_size}")

        # Calculate left crop coordinates
        left = 0
        upper = (height - crop_size) // 2
        right = crop_size
        lower = upper + crop_size

        return img.crop((left, upper, right, lower))