# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image

# import dnnlib

from quant.edm2.misc import EasyDict
from quant.edm2 import distributed as dist

warnings.filterwarnings("ignore", "`resume_download` is deprecated")

# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


# ----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)


def generate_images(
    net,  # Main network. Path, URL, or torch.nn.Module.
    encoder=None,  # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir=None,  # Where to save the output images. None = do not save.
    subdirs=False,  # Create subdirectory for every 1000 seeds?
    seeds=range(16, 24),  # List of random seeds.
    class_idx=None,  # Class label. None = select randomly.
    max_batch_size=32,  # Maximum batch size for the diffusion model.
    encoder_batch_size=4,  # Maximum batch size for the encoder. None = default.
    verbose=True,  # Enable status prints?
    device=torch.device("cuda"),  # Which compute device to use.
    **sampler_kwargs,  # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    pipe = ###
    assert net is not None

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f"Setting up {type(encoder).__name__}...")
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, "batch_size"):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = (
        max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1)
        * dist.get_world_size()
    )
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[
        dist.get_rank() :: dist.get_world_size()
    ]
    if verbose:
        dist.print0(f"Generating {len(seeds)} images...")

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(
                    images=None,
                    labels=None,
                    noise=None,
                    batch_idx=batch_idx,
                    num_batches=len(rank_batches),
                    indices=indices,
                )
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn(
                        [
                            len(r.seeds),
                            net.img_channels,
                            net.img_resolution,
                            net.img_resolution,
                        ],
                        device=device,
                    )
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[
                            rnd.randint(
                                net.label_dim, size=[len(r.seeds)], device=device
                            )
                        ]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = None
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(
                            r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()
                        ):
                            image_dir = (
                                os.path.join(outdir, f"{seed//1000*1000:06d}")
                                if subdirs
                                else outdir
                            )
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, "RGB").save(
                                os.path.join(image_dir, f"{seed:06d}.png")
                            )

                # Yield results.
                torch.distributed.barrier()  # keep the ranks in sync
                yield r

    return ImageIterable()


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------
# Command line interface.


@click.command()
@click.option(
    "--preset", help="Configuration preset", metavar="STR", type=str, default=None
)
@click.option(
    "--net", help="Network pickle filename", metavar="PATH|URL", type=str, default=None
)
@click.option(
    "--gnet",
    help="Reference network for guidance",
    metavar="PATH|URL",
    type=str,
    default=None,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--subdirs", help="Create subdirectory for every 1000 seeds", is_flag=True
)
@click.option(
    "--seeds",
    help="List of random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="16-19",
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    help="Class label  [default: random]",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of sampling steps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=0.002,
    show_default=True,
)
@click.option(
    "--sigma_max",
    help="Highest noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=80,
    show_default=True,
)
@click.option(
    "--rho",
    help="Time step exponent",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=7,
    show_default=True,
)
@click.option(
    "--guidance",
    help="Guidance strength  [default: 1; no guidance]",
    metavar="FLOAT",
    type=float,
    default=None,
)
@click.option(
    "--S_churn",
    "S_churn",
    help="Stochasticity strength",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_min",
    "S_min",
    help="Stoch. min noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_max",
    "S_max",
    help="Stoch. max noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default="inf",
    show_default=True,
)
@click.option(
    "--S_noise",
    "S_noise",
    help="Stoch. noise inflation",
    metavar="FLOAT",
    type=float,
    default=1,
    show_default=True,
)
def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = EasyDict(opts)

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit="batch", disable=(dist.get_rank() != 0)):
        pass


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

# ----------------------------------------------------------------------------
