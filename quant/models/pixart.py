import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

from quant.models.base import Base


class PixArt(Base):
    def __init__(self, device="cuda"):
        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
            use_safetensors=True,
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=self.weight_dtype,
            use_safetensors=True,
        ).to(device)


if __name__ == "__main__":
    """
    python -m quant.models.pixart "A beautiful sunset"
    """
    from fire import Fire

    model = PixArt()
    Fire(model.gen)
