from torch.utils.data import DataLoader

from safety_checker.src.image_folder import ImageFolder

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/dataloader_utils.py


def collate_fn(batch):
    """Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(
    data_dir,
    transform,
    batch_size: int = 128,
    num_workers: int = 2,
    collate_fn=collate_fn,
):
    """Get dataloader for the images in the data_dir."""
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
