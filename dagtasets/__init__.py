"""A collection of datasets for document and text analysis curated for pytorch.
"""

from dagtasets import util
from dagtasets.dag_dataset import DagDataset
from dagtasets.hwi2017 import HWI2017
from dagtasets.mjsynth import mjsynth_color, mjsynth_color_pad, mjsynth_color_scale, mjsynth_gray, mjsynth_gray_pad, \
    mjsynth_gray_scale, MjSynth, RandomPadAndNormalise
from dagtasets.siamese import SiameseDs
from dagtasets.wi2013 import WI2013
from util import mkdir_p, load_image_float, save_image_float, resumable_download

util.check_os_dependencies()

__all__ = [SiameseDs, WI2013, DagDataset, HWI2017, mkdir_p, load_image_float, save_image_float, resumable_download,
           mjsynth_color, mjsynth_color_pad, mjsynth_color_scale, mjsynth_gray, mjsynth_gray_pad, mjsynth_gray_scale,
           MjSynth, RandomPadAndNormalise]
