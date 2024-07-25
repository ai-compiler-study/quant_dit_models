import torch


class Base:
    hf_name = ""
    variant = None
    pipeline_cls = None
    weight_dtype = torch.float16

    def __init__(self):
        assert self.hf_name, "hf_name not defined"
        assert self.pipeline_cls, "pipeline_cls not defined"

        self.pipe = self.pipeline_cls.from_pretrained(
            self.hf_name, torch_dtype=self.weight_dtype, variant=self.variant
        ).to("cuda")

    def gen(
        self,
        prompt,
        neg_prompt="",
        guidance_scale=5.0,
        num_inference_steps=50,
        seed=66,
    ):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(self.pipe.device).manual_seed(seed),
        ).images[0]
        return image
