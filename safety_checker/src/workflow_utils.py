import numpy as np
import torch
from src.device_utils import get_device
from src.safety_checker_model import (
    get_safety_checker_model,
    get_safety_checker_processor,
)
from src.safety_checker_utils import detect_bad_concepts
from tqdm.auto import tqdm

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/feature_utils.py


def apply_workflow(img_loader, batch_size, verbose=True):
    device = get_device()
    processor = get_safety_checker_processor()
    safety_checker_model = get_safety_checker_model()

    aggregate = []
    sample_fnames = []

    with torch.no_grad():
        for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
            if verbose:
                print(f"\nExtraction of batch n°{ii}.\n")
            clip_input = torch.tensor(np.array(processor(imgs).pixel_values)).to(device)
            bad_concepts = detect_bad_concepts(safety_checker_model, clip_input)

            aggregate += bad_concepts
            sample_fnames += [
                img_loader.dataset.samples[ii * batch_size + jj]
                for jj in range(len(bad_concepts))
            ]

    return aggregate, sample_fnames
