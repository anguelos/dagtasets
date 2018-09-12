"""A collection of datasets for document and text analysis curated for pytorch.
"""

import util
from dagtasets.dag_dataset import DagDataset
from dagtasets.hwi2017 import HWI2017
from dagtasets.mjsynth import mjsynth_color, mjsynth_color_pad, mjsynth_color_scale, mjsynth_gray, mjsynth_gray_pad, \
    mjsynth_gray_pad_height, mjsynth_gray_scale, MjSynthWS, MjSynthTranscription, RandomPadAndNormalise
from dagtasets.siamese import SiameseDs, BatchMiningSampler
from dagtasets.wi2013 import WI2013
from util import mkdir_p, load_image_float, save_image_float, resumable_download, get_variable_lenght_dataloader

util.check_os_dependencies()

__all__ = [SiameseDs, BatchMiningSampler, WI2013, DagDataset, HWI2017, mkdir_p, load_image_float, save_image_float,
           resumable_download, mjsynth_color, mjsynth_color_pad, mjsynth_color_scale, mjsynth_gray, mjsynth_gray_pad,
           mjsynth_gray_pad_height, mjsynth_gray_scale, MjSynthWS, MjSynthTranscription, RandomPadAndNormalise]
