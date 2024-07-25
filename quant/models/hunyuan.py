from diffusers import HunyuanDiTPipeline

from quant.models.base import Base


class Hunyuan(Base):
    hf_name = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    pipeline_cls = HunyuanDiTPipeline


if __name__ == "__main__":
    """
    python -m quant.models.hunyuan "A beautiful sunset"
    """
    from fire import Fire

    model = Hunyuan()
    Fire(model.gen)
