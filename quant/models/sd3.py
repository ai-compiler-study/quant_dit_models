import torch
from diffusers import StableDiffusion3Pipeline

from quant.models.base import Base


class SD3(Base):
    hf_name = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipeline_cls = StableDiffusion3Pipeline


if __name__ == "__main__":
    """
    python -m quant.models.sd3 "A beautiful sunset"
    """
    from fire import Fire

    model = SD3()
    Fire(model.gen)
