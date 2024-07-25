from diffusers import KolorsPipeline

from .base import Base


class Kolors(Base):
    hf_name = "Kwai-Kolors/Kolors-diffusers"
    pipeline_cls = KolorsPipeline
    variant = "fp16"


if __name__ == "__main__":
    """
    python -m quant.models.kolors "A beautiful sunset"
    """
    from fire import Fire

    model = Kolors()
    Fire(model.gen)
