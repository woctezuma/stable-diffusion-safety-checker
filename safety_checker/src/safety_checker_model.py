import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from src.device_utils import get_device
from transformers import AutoFeatureExtractor

MODEL_NAME = "CompVis/stable-diffusion-safety-checker"

# Reference:
# https://huggingface.co/CompVis/stable-diffusion-safety-checker


def get_safety_checker_processor(dtype=torch.float16):
    return AutoFeatureExtractor.from_pretrained(
        MODEL_NAME,
        device=get_device(),
        dtype=dtype,
    )


def get_safety_checker_model():
    return StableDiffusionSafetyChecker.from_pretrained(MODEL_NAME).to(get_device())
