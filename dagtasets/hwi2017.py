#!/usr/bin/env python

import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import re
from StringIO import StringIO
from torchvision import transforms
from commands import getoutput as shell_stdout
from zipfile import ZipFile
import glob
from PIL import Image
import numpy as np

from .util import resumable_download

transform_color = transforms.Compose([
    transforms.RandomCrop((1024,512), padding=1, pad_if_needed=True),
    #transforms.Resize((512,192)),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_target = transforms.Compose([
])

def read_rar_file(rarstream):
    names=rarstream.namelist()
    sample_list=[]
    for name in names:
        img=Image.open(StringIO(rarstream.read(rarstream.getinfo(name)))).copy()
        [writer_id,sample_id] =[int(s) for s in name[:name.find(".")].split("_")]
        language_id=(sample_id-1)/2 # icdar 2013: english 1,2 and greek 3,4
        sample_list.append((img,(writer_id,language_id,sample_id)))
    return sample_list



class HWI2017(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = {
        'train':'ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-training-color.zip',
        'test':'ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-dataset-flat-color.zip'
    }
    zip_folder='zips'
    img_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def _check_exists(self,partition=None):
        if partition is None:
            return all([self._check_exists(p) for p in self.urls.keys()])
        else:
            try:
                open(self.root + "/" + partition + ".marker", "r").read()
                return True
            except IOError:
                return False

    def _set_exists(self,partition,val=True):
        if val:
            open(self.root + "/" + partition + ".marker", "w").write("1")
        else:
            os.unlink(self.root + "/" + partition + ".marker")

    def _load_dir(self,partition):
        img_list=glob.glob(self.root+"/img/"+partition+"/*.jpg")

    def __init__(self, root, train=True, transform=transform_color, target_transform=transform_target, download=False,
                 output_class="writer"):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download_if_needed()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            partition='train'
        else:
            partition='test'

        img_list = glob.glob(self.root+"/img/"+partition+"/*.jpg")
        img_list += glob.glob(self.root + "/img/" + partition + "/*.png")
        self.image_paths=np.array(img_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name = self.image_paths[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_name)

        if self.transform is not    None:
            img = self.transform(img)
        _, filename= os.path.split(img_name)
        target = int(re.findall("^[0-9]+",filename)[0])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.image_paths.shape[0]

    def download_if_needed(self):
        for partition,url in self.urls.items():
            if not self._check_exists(partition):
                zip_path = resumable_download(url,self.root + "/zips/")
                #mkdir_p(self.root + "/zips/")
                #download_cmd = 'wget --directory-prefix=%s -c %s' % (self.root+"/zips/", url)
                #shell_stdout(download_cmd)
                #zipfile=self.root+"/zips/"+url.split("/")[-1]
                archive = ZipFile(zip_path)
                dir_names=[n for n in archive.namelist() if n[-1]=="/"]
                assert len(dir_names) == 1
                print 'Extracting %s ... '%zip_path,
                archive.extractall(self.root+"/img/")
                shell_stdout("mv "+self.root+"/img/"+dir_names[0]+" "+self.root+"/img/"+partition)
                print 'done'
                self._set_exists(partition,True)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__=="__main__":
    trds = WI2013('./data/wi2013', train=True, transform=None, target_transform=None, download=True)
    tstds = WI2013('./data/wi2013', train=False, transform=None, target_transform=None, download=True)
