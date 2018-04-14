"""A collection of datasets for document and text analysis curated for pytorch.
"""

from dagtasets.siamese import SiameseDs
from dagtasets.wi2013 import  WI2013
from dagtasets.dag_dataset import DagDataset
from dagtasets.hwi2017 import HWI2017
from util import  mkdir_p,load_image_float,save_image_float,resumable_download
from dagtasets import util
util.check_os_dependencies()

__all__ = [SiameseDs,WI2013,DagDataset,HWI2017,mkdir_p,load_image_float,save_image_float,resumable_download]