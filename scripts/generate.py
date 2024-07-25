import os
import os.path as osp
from tqdm import tqdm

from quant.models import MODELS
from quant.constants import ASSET_DIR


def main(name, prompt_path, dst_root_dir=ASSET_DIR, **kwargs):
    dst_dir = osp.join(dst_root_dir, name)
    os.makedirs(dst_dir, exist_ok=True)
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
        prompts.sort()
    model = MODELS[name]()
    for idx, prompt in enumerate(tqdm(prompts)):
        dst_path = osp.join(dst_dir, f"{idx:04d}.jpg")
        if osp.exists(dst_path):
            continue
        img = model.gen(prompt, **kwargs)
        img.save(dst_path)


if __name__ == "__main__":
    """
    python ./scripts/generate.py hunyuan ./assets/prompts.txt
    python ./scripts/generate.py hunyuan_distill ./assets/prompts.txt --num_inference_steps=25
    """
    from fire import Fire

    Fire(main)
