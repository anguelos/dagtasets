from __future__ import print_function

import torch

import commands
import os
import re
from collections import defaultdict

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from .util import RandomPadAndNormalise, resumable_download, mkdir_p, extract
import shutil
import PIL
from types import MethodType
import lm_util

# Composite Transform scaling the image as the original dataset was used.
mjsynth_gray_scale = transforms.Compose([
    transforms.Resize(32, 104),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Composite Transform scaling the image as the original dataset was used.
mjsynth_color_scale = transforms.Compose([
    transforms.Resize(32, 104),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Composite Transform leaving the image at its size.
# The collate function has to account for the different sizes.
mjsynth_gray = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Composite Transform leaving the image at its size.
# The collate function has to account for the different sizes.
mjsynth_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Composite Transform padding the image with gaussian noise and
# then standarizing pixel values.
mjsynth_gray_pad = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    RandomPadAndNormalise((32, 256))
])

# Composite Transform padding the image with gaussian noise and
# then standarizing pixel values.
mjsynth_gray_pad_height = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    RandomPadAndNormalise((32, -1))
])


# Composite Transform padding the image with gaussian noise and
# then standarizing pixel values.
mjsynth_color_pad = transforms.Compose([
    transforms.ToTensor(),
    RandomPadAndNormalise((32, 256))
])


class _MjSynth(data.Dataset):
    """Sythetic word-image dataset.

    References:
    @article{Jaderberg14c,
      title={Synthetic Data and Artificial Neural Networks for Natural Scene
      Text Recognition},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1406.2227},
      year={2014}
    }

    @article{Jaderberg14d,
      title={Reading Text in the Wild with Convolutional Neural Networks},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1412.1842},
      year={2014}
    }
    :param root (string): Root directory of dataset where everything will be
        stored.
    :param transform (callable): A functor taking PIL images as input and
        returning 3D pytorch tensors of dimensions
        [channels x height x width].
    :param train (bool, optional): If True, it will load samples from
        annotations_train.txt, other wise annotations_val will be used.
    :param output: A string that is either "class" or "transcription". If
        "class", the target of each sample is an integer, otherwise the
        target is a unicode string.
    :param download (bool, optional): If true, downloads the dataset from
        the internet and puts it in root directory. If dataset is already
        downloaded at the same location, it is not downloaded again. The
        total size of the Archive is ~9GB.
    :param target_transform (callable, optional): A function/transform that
        takes in the target and transforms it to what a loss function would
        expect.
    :param remove_archive(boolean): If True only the images will be stored
        and the tar.gz archive is erased once no longer needed.
    :param encoder: If None the target of each sample is an integer.
        Otherwise it must be an lm_util.Encoder and the target of each
        sample will be a sequence of integers.

    Example: ds = dagtasets.MjSynth("/tmp/",dagtasets.mjsynth_gray,download=True)
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz'

    # TODO (anguelos): save to pickle.

    def _check_exists(self):
        try:
            if open(os.path.join(self.root, ".marker"), "r").read() == "1":
                return True
            return False
        except IOError:
            return False

    def download_if_needed(self):
        if not self._check_exists():
            zip_dir = os.path.join(self.root, self.zip_folder)
            mkdir_p(zip_dir)
            resumable_download(MjSynth.url,zip_dir)
            self._extract()
            self._set_exists()
        else:
            print ("Dataset found. Not downloading.")

    def _set_exists(self, val=True):
        if val:
            open(os.path.join(self.root, ".marker"), "w").write("1")
        else:
            os.unlink(os.path.join(self.root, ".marker"))

    def _extract(self):
        mj_subdirs = ["mnt", "ramdisk", "max", "90kDICT32px"]
        dest_dir = os.path.join(self.root, self.img_folder)
        archive_path = os.path.join(self.root, self.zip_folder,
                                    "mjsynth.tar.gz")
        tmp_folder = os.path.join(self.root, self.tmp_folder)
        mjsynth_root = os.path.join(tmp_folder, *mj_subdirs)

        if dest_dir[-1] == "/":
            dest_dir = dest_dir[:-1]
        extract(archive_path,tmp_folder)
        dest_parent_dir = os.path.dirname(dest_dir)
        mkdir_p(dest_parent_dir)
        shutil.move(mjsynth_root,dest_dir)
        if self.remove_archive:
            os.unlink(archive_path)
        for k in reversed(range(len(mj_subdirs))):
            os.rmdir(os.path.join(tmp_folder, *mj_subdirs[:k]))
        self._set_exists()

    def _load_dir(self):
        if self.train:
            file_list = os.path.join(self.root, self.img_folder,
                                     "annotation_train.txt")
        else:
            file_list = os.path.join(self.root, self.img_folder,
                                     "annotation_val.txt")
        lines = [line.split(" ") for line in
                 open(file_list).read().strip().split("\n")]

        self.class_ids = np.array([int(line[1]) for line in lines])
        self.transcriptions = np.array(
            [unicode(line[0].split("_")[-1]) for line in lines])
        #samples_by_class = defaultdict(lambda: [])
        #for k in range(len(lines)):
        #    samples_by_class[int(lines[k][1])].append(k)
        #class2idx = np.empty(max(samples_by_class.keys()) + 1, dtype=object)
        #for k, v in samples_by_class.items():
        #    class2idx[k] = np.array(v)
        #self.class2idx = class2idx
        self.filenames = np.array([line[0] for line in lines])

    def __init__(self, root, transform, train=True,
                 target_transform=None, download=False,
                 remove_archive=True):
        self.root = os.path.expanduser(root)
        self.zip_folder = 'zips'
        self.img_folder = 'raw'
        self.tmp_folder = 'tmp'
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.remove_archive = remove_archive

        if download:
            self.download_if_needed()
        else:

            if not self._check_exists():
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')
        self._load_dir()


    def __len__(self):
        return self.class_ids.shape[0]




class MjSynthWS(_MjSynth):
    """Sythetic word-image dataset.

    References:
    @article{Jaderberg14c,
      title={Synthetic Data and Artificial Neural Networks for Natural Scene
      Text Recognition},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1406.2227},
      year={2014}
    }

    @article{Jaderberg14d,
      title={Reading Text in the Wild with Convolutional Neural Networks},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1412.1842},
      year={2014}
    }
    :param root (string): Root directory of dataset where everything will be
        stored.
    :param transform (callable): A functor taking PIL images as input and
        returning 3D pytorch tensors of dimensions
        [channels x height x width].
    :param train (bool, optional): If True, it will load samples from
        annotations_train.txt, other wise annotations_val will be used.
    :param output: A string that is either "class" or "transcription". If
        "class", the target of each sample is an integer, otherwise the
        target is a unicode string.
    :param download (bool, optional): If true, downloads the dataset from
        the internet and puts it in root directory. If dataset is already
        downloaded at the same location, it is not downloaded again. The
        total size of the Archive is ~9GB.
    :param target_transform (callable, optional): A function/transform that
        takes in the target and transforms it to what a loss function would
        expect.
    :param remove_archive(boolean): If True only the images will be stored
        and the tar.gz archive is erased once no longer needed.
    :param encoder: If None the target of each sample is an integer.
        Otherwise it must be an lm_util.Encoder and the target of each
        sample will be a sequence of integers.

    Example: ds = dagtasets.MjSynth("/tmp/",dagtasets.mjsynth_gray,download=True)
    """

    def __init__(self, root, transform=mjsynth_gray_pad, train=True,
                 target_transform=None, download=False,
                 remove_archive=True):
        super(MjSynthWS, self).__init__(root=root, transform=transform, train=train,
                 target_transform=target_transform, download=download,
                 remove_archive=remove_archive)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = os.path.join(self.root, self.img_folder,
                                self.filenames[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            image = Image.open(img_path)
        except:
            image = PIL.Image.fromarray((np.random.random([400, 400, 3]) * 255).astype("uint8"))
            print("Failed to convert ", img_path)

        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                image = PIL.Image.fromarray((np.random.random([400, 400, 3]) * 255).astype("uint8"))
                image = self.transform(image)
                print("Failed to convert ",img_path)

        _, filename = os.path.split(img_path)
        target = int(re.findall("^[0-9]+", filename)[0])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class MjSynthTranscription(_MjSynth):
    """Sythetic word-image dataset.

    References:
    @article{Jaderberg14c,
      title={Synthetic Data and Artificial Neural Networks for Natural Scene
      Text Recognition},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1406.2227},
      year={2014}
    }

    @article{Jaderberg14d,
      title={Reading Text in the Wild with Convolutional Neural Networks},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1412.1842},
      year={2014}
    }
    :param root (string): Root directory of dataset where everything will be
        stored.
    :param transform (callable): A functor taking PIL images as input and
        returning 3D pytorch tensors of dimensions
        [channels x height x width].
    :param train (bool, optional): If True, it will load samples from
        annotations_train.txt, other wise annotations_val will be used.
    :param output: A string that is either "class" or "transcription". If
        "class", the target of each sample is an integer, otherwise the
        target is a unicode string.
    :param download (bool, optional): If true, downloads the dataset from
        the internet and puts it in root directory. If dataset is already
        downloaded at the same location, it is not downloaded again. The
        total size of the Archive is ~9GB.
    :param target_transform (callable, optional): A function/transform that
        takes in the target and transforms it to what a loss function would
        expect.
    :param remove_archive(boolean): If True only the images will be stored
        and the tar.gz archive is erased once no longer needed.
    :param encoder: An lm_util.Encoder (having a .encode method) that maps
        strings to integer sequences.

    Example: ds = dagtasets.MjSynth("/tmp/",dagtasets.mjsynth_gray,download=True)
    """

    def __init__(self, root, transform=mjsynth_gray_pad_height, train=True,
                 target_transform=None, download=False,
                 remove_archive=True, encoder=lm_util.letter_encoder):
        super(MjSynthTranscription, self).__init__(root=root, transform=transform, train=train,
                                    target_transform=target_transform, download=download,
                                    remove_archive=remove_archive)
        self.encoder=encoder

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a numpy vector of integers.
        """
        img_path = os.path.join(self.root, self.img_folder,
                                self.filenames[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            image = Image.open(img_path)
        except:
            image = PIL.Image.fromarray((np.random.random([400, 400, 3]) * 255).astype("uint8"))
            print("Failed to convert ", img_path)

        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                image = PIL.Image.fromarray((np.random.random([400, 400, 3]) * 255).astype("uint8"))
                image = self.transform(image)
                print("Failed to convert ",img_path)

        _, filename = os.path.split(img_path)
        target = self.encoder.encode(filename.split("_")[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.IntTensor(target)
        return image, target
