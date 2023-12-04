from torchvision import transforms

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/transform_utils.py


def get_target_image_size(resize_size=256, keep_ratio=True):
    return resize_size if keep_ratio else (resize_size, resize_size)


def get_transform(
    resize_size=256,
    keep_ratio=True,
    interpolation=transforms.InterpolationMode.BICUBIC,
):
    transforms_list = [
        transforms.Resize(
            get_target_image_size(resize_size, keep_ratio),
            interpolation=interpolation,
        ),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transforms_list)
