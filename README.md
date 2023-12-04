# Stable Diffusion Safety Checker

The goal of this repository is to run the [safety checker][huggingface-safety-checker] from [Stable Diffusion][huggingface-stable-diffusion].

## Requirements

- Install the latest version of [Python 3][python-download-url].
- Install the required packages:

```bash
!pip install git+https://github.com/woctezuma/stable-diffusion-safety-checker.git
```

## Usage

- Run the main script with:

```bash
!python -m safety_checker.check_safety -h
```

- Alternatively, run [`safety_checker.ipynb`][colab-notebook-safety-checker].
[![Open In Colab][colab-badge]][colab-notebook-safety-checker]

## Example

Download the `balloon` image dataset.
```bash
fname = "balloon_dataset.zip"
!curl -OL https://github.com/matterport/Mask_RCNN/releases/download/v2.1/{fname}
!unzip -q {fname}
```

Run the script:
```bash
!python -m safety_checker.check_safety \
 --input balloon \
 --batch 8 \
 --resize 256 \
 --keep-ratio \
 --output bad_concepts.json \
 --verbose
```

Check the results:

```python
import json

from pathlib import Path

with Path("bad_concepts.json").open(encoding='utf8') as f:
  results = json.load(f)
```

The IDs of the "bad concepts" are clarified on [this page][bad-concepts] hosted by LAION-AI.

## References

- [`feature-extractor`][feature-extractor]: similar code to extract image features,
- [`discord-members-metadata`][data-discord]: profiles pictures scraped from a specific Discord guild's members.

<!-- Definitions -->

[huggingface-safety-checker]: <https://huggingface.co/CompVis/stable-diffusion-safety-checker>
[huggingface-stable-diffusion]: <https://huggingface.co/CompVis/stable-diffusion>

[python-download-url]: <https://www.python.org/downloads/>
[bad-concepts]: <https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/safety_settings.yml>

[feature-extractor]: <https://github.com/woctezuma/feature-extractor>
[data-discord]: <https://github.com/woctezuma/discord-members-metadata>

[colab-notebook-safety-checker]: <https://colab.research.google.com/github/woctezuma/stable-diffusion-safety-checker/blob/main/safety_checker.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
