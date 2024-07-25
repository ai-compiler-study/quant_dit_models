from quant.models.sd3 import SD3
from quant.models.pixart import PixArt
from quant.models.kolors import Kolors
from quant.models.hunyuan import Hunyuan
from quant.models.hunyuan_distill import HunyuanDistill

MODELS = {
    "sd3": SD3,
    "pixart": PixArt,
    "kolors": Kolors,
    "hunyuan": Hunyuan,
    "hunyuan_distill": HunyuanDistill,
}
