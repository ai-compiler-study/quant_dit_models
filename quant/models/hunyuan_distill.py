from diffusers import HunyuanDiTPipeline

from quant.models.base import Base


class HunyuanDistill(Base):
    hf_name = "Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled"
    pipeline_cls = HunyuanDiTPipeline


if __name__ == "__main__":
    """
    python -m quant.models.hunyuan_distill "A beautiful sunset"
    """
    from fire import Fire

    model = HunyuanDistill()
    Fire(model.gen)
