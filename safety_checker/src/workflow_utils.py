import numpy as np
import torch
from tqdm.auto import tqdm

from safety_checker.src.device_utils import get_device
from safety_checker.src.safety_checker_model import (
    get_safety_checker_model,
    get_safety_checker_processor,
)
from safety_checker.src.safety_checker_utils import detect_bad_concepts

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/feature_utils.py


def apply_workflow(img_loader, batch_size, verbose=True):
    device = get_device()
    processor = get_safety_checker_processor()
    safety_checker_model = get_safety_checker_model()

    aggregate = []
    scores = []
    sample_fnames = []

    with torch.no_grad():
        for ii, imgs in enumerate(tqdm(img_loader)):
            if verbose:
                print(f"\nExtraction of batch n°{ii}.\n")
            clip_input = torch.tensor(np.array(processor(imgs).pixel_values)).to(device)
            bad_concepts, bad_concepts_scores = detect_bad_concepts(
                safety_checker_model,
                clip_input,
            )

            aggregate += bad_concepts
            scores.append(bad_concepts_scores)
            sample_fnames += [
                img_loader.dataset.samples[ii * batch_size + jj]
                for jj in range(len(bad_concepts))
            ]

    scores = torch.concat(scores, dim=0)

    return aggregate, scores, sample_fnames
