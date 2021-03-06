from __future__ import print_function

import math
import os
import sys
import zipfile

import rarfile
import torch
import torchvision
from PIL import Image

from .util import resumable_download, shell_stdout
from .DiamondSquare import diamond_square
from io import BytesIO as FileReadWrapper


class RandomPlasma(object):
    def __init__(self,occurence_prob=.3, mask_gt=False, quantile_min=.05,quantile_max=.5, scale_factor=2.0,roughness_min=.1,
                 roughness_max=.9,low_quantile_prob=.5,max_bg_range=.7):
        self.occurence_prob=occurence_prob
        self.scale_factor=scale_factor
        self.roughness_min=roughness_min
        self.roughness_max=roughness_max
        self.quantile_min=quantile_min
        self.quantile_max=quantile_max
        self.low_quantile_prob=low_quantile_prob
        self.max_bg_range=max_bg_range
        self.mask_gt=mask_gt


    def __call__(self,input_img,gt,original_img=None):
        if original_img is None:
            original_img = torch.ones_like(input_img[0,:,:])
        if torch.rand(1).item()>self.occurence_prob:
            return input_img, gt, original_img
        quantile=torch.rand(1).item()*(self.quantile_max-self.quantile_min)+self.quantile_min
        roughness=torch.rand(1).item()*(self.roughness_max-self.roughness_max)+self.roughness_min
        remove=torch.rand(1).item()>self.low_quantile_prob
        min_bg=torch.rand(1).item()
        min_bg,max_bg=sorted([min_bg,min_bg+torch.rand(1).item()*self.max_bg_range-self.max_bg_range/2])
        min_bg=max(min_bg,0.0)
        max_bg = min(max_bg, 1.0)
        _,width,height = input_img.size()
        plasma_width,plasma_height=int(math.ceil(width/self.scale_factor)),int(math.ceil(height/self.scale_factor))

        #creating small plasma for speed
        plasma=diamond_square([plasma_width,plasma_height],0,10,roughness)
        plasma=torch.Tensor(plasma)
        _min=plasma.min().item()
        plasma = (max_bg-min_bg)*((plasma-_min)/(plasma.max()-_min))+min_bg

        #growing plasma to image size
        plasma=torch.nn.functional.interpolate(torch.Tensor(plasma).unsqueeze(dim=0).unsqueeze(dim=0),
                                               scale_factor=[self.scale_factor,self.scale_factor],
                                               mode='bilinear', align_corners=True)[0,0,:width,:height]
        if remove:
            quantile_idx=int(plasma.view(-1).size()[0]*(1-quantile))
            thr=torch.kthvalue(plasma.view(-1), quantile_idx)[0].item()
            mask=plasma<thr
        else:
            quantile_idx = int(plasma.view(-1).size()[0]*(quantile))
            thr=torch.kthvalue(plasma.view(-1), quantile_idx)[0].item()
            mask=plasma>thr
        mask=mask.float()
        original_img = original_img * mask
        input_img=input_img[0,:,:]*original_img+plasma*(1-original_img)
        if self.mask_gt:
            gt = mask * gt[1,:,:]
            gt = torch.cat([1 - gt.unsqueeze(dim=0), gt.unsqueeze(dim=0)], dim=0)
        input_img = torch.cat([input_img.unsqueeze(dim=0),1-input_img.unsqueeze(dim=0)],dim=0)
        return input_img, gt, original_img


class RandomCropTo(object):
    """Functor scaling and cropping pairs of tensor images.

    """

    def __init__(self, minimum_size=[512, 512], pad_if_needed=True, scale_range=(1.0, 1.0)):
        self.minimum_width, self.minimum_height = minimum_size
        self.pad_if_needed = pad_if_needed
        self.scale_range = scale_range

    def __call__(self, input_img, gt):
        _, width, height = input_img.size()
        if self.scale_range != (1.0, 1.0):
            scale = self.scale_range[0] + float(torch.rand(1)) * (self.scale_range[1] - self.scale_range[0])
            width = int(math.round(width * scale))
            height = int(math.round(height * scale))
            input_img = torch.nn.functional.interpolate(input_img.unsqueeze(dim=0), (width, height), mode="bilinear",align_corners=True)
            gt = torch.nn.functional.interpolate(gt.unsqueeze(dim=0), (width, height), mode="nearest")
            input_img, gt = torch.squeeze(input_img, 0), torch.squeeze(gt, 0)
        if (width < self.minimum_width or height < self.minimum_height) and self.pad_if_needed:
            if width < self.minimum_width:
                x_needed = 1+self.minimum_width - width
            else:
                x_needed = 0

            if height < self.minimum_height:
                y_needed = 1+self.minimum_height - height
            else:
                y_needed = 0
            original_img=torch.ones_like(input_img)
            input_img = torch.nn.functional.pad(input_img, (int(y_needed / 2), int(y_needed - y_needed / 2),
                                                            int(x_needed / 2), int(x_needed - x_needed / 2)))
            gt = torch.nn.functional.pad(gt[1,:,:], (int(y_needed / 2), int(y_needed - y_needed / 2),
                                              int(x_needed / 2), int(x_needed - x_needed / 2)))
            gt = torch.cat([1-gt.unsqueeze(dim=0),gt.unsqueeze(dim=0)],dim=0)
            original_img = torch.nn.functional.pad(original_img, (int(y_needed / 2), int(y_needed - y_needed / 2),
                                              int(x_needed / 2), int(x_needed - x_needed / 2)))
        else:
            original_img = torch.ones_like(input_img)
        #print(width,height,input_img.size())
        max_left = max(width - self.minimum_width, 1)
        max_top = max(height - self.minimum_height, 1)
        #print("Maxleft:",max_left,"   Maxtop:",max_top)
        left = torch.randint(low=0, high=max_left, size=[1])[0]
        top = torch.randint(low=0, high=max_top, size=[1])[0]
        right = left + self.minimum_width
        bottom = top + self.minimum_height
        #print("LTRB",left,top,right,bottom)
        #print(input_img.size(), gt.size())
        input_img, gt, original_img = input_img[:, left:right, top:bottom], gt[:, left:right, top:bottom], original_img[:, left:right, top:bottom]

        #print(input_img.size(), gt.size())
        return input_img, gt,original_img[0,:,:]


dibco_transform_gray_train = torchvision.transforms.Compose([
    #torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x: torch.cat([x, 1 - x])
])

dibco_transform_color_train = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
    torchvision.transforms.ToTensor()
])

dibco_transform_gray_inference = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x: torch.cat([x, 1 - x])
])

dibco_transform_color_inference = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])


class Dibco:
    """Provides one or more of the DIBCO datasets.

    Other than standard torchvision augmentations, Augmentation

    Os dependencies: Other than python packages, unrar and arepack CLI tools must be installed.
    In Ubuntu they can be installed with: sudo apt install unrar atool p7zip-full
    """
    urls = {
        "2009_HW": ["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar",
                    "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar"],

        "2009_P": ["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar",
                   "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar"],

        "2010": ["http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar",
                 "http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar"],

        "2011_P": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar"],
        "2011_HW": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar"],

        "2012": ["http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/H-DIBCO2012-dataset.rar"],

        "2013": ["http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar"],

        "2014": ["http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar",
                 "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar"],
        "2016": ["https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
                 "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip"],
        "2017": ["https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z",
                 "https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z"],
        "2018": ["http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018_Dataset.zip",
                 "http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018-GT.zip"]
    }

    @staticmethod
    def load_single_rar_stream(compressed_stream):
        def name2img(x):
            return Image.open(FileReadWrapper((compressed_stream.read(compressed_stream.getinfo(x))))).copy()

        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: name2img(n) for n in compressed_stream.namelist() if
                 "." in n and "_" in n.split("/")[-1] and "skelGT" not in n}
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: name2img(n) for n in compressed_stream.namelist() if
                 "." in n and "_" not in n.split("/")[-1]}
        assert set(id2gt.keys()) == set(id2in.keys())
        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def load_two_rar_stream(input_compressed_stream, gt_compressed_stream, ):
        def in_name2img(x):
            return Image.open(
                FileReadWrapper((input_compressed_stream.read(input_compressed_stream.getinfo(x))))).copy()

        def gt_name2img(x):
            return Image.open(FileReadWrapper((gt_compressed_stream.read(gt_compressed_stream.getinfo(x))))).copy()

        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: in_name2img(n) for n in
                 input_compressed_stream.namelist() if "." in n and "_" not in n.split("/")[-1]}
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: gt_name2img(n) for n in gt_compressed_stream.namelist()
                 if "." in n and "skelGT" not in n and not n.endswith(".dat")}
        assert set(id2gt.keys()) == set(id2in.keys())

        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}


    @staticmethod
    def Dibco2009(**kwargs):
        kwargs["partitions"] = ["2009_HW", "2009_P"]
        return Dibco(**kwargs)


    @staticmethod
    def Dibco2010(**kwargs):
        kwargs["partitions"] = ["2010"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2011(**kwargs):
        kwargs["partitions"] = ["2011_P","2011_HW"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2012(**kwargs):
        kwargs["partitions"] = ["2012"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2013(**kwargs):
        kwargs["partitions"] = ["2013"]
        return Dibco(**kwargs)

    def __init__(self, partitions=["2009_HW", "2009_P"], crop_sz=[512, 512], root="/tmp/dibco", train=True,
                 scale_range=None,
                 input_transform=dibco_transform_gray_train, gt_transform=dibco_transform_gray_train,max_plasma_quantile=.5, max_plasma_roughness=.7,mask_gt=False):
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.root = root
        if (crop_sz is not None or scale_range is not None) and train:
            if scale_range is None:
                scale_range = (1.0, 1.0)
            self.crop = RandomCropTo(crop_sz, scale_range=scale_range)
        else:
            self.crop = lambda x, y: (x, y, torch.ones_like(x))
        if (max_plasma_quantile>0) and train:
            self.plasma=RandomPlasma(occurence_prob=1.0,roughness_max=max_plasma_roughness,mask_gt=mask_gt,quantile_max=max_plasma_quantile)
        else:
            self.plasma = lambda x, y, z: (x, y, z)
        data = {}
        for partition in partitions:
            for url in Dibco.urls[partition]:
                archive_fname = root + "/" + url.split("/")[-1]
                if not os.path.isfile(archive_fname):
                    resumable_download(url, root)
                else:
                    print(archive_fname," found in cache.")
                if url.endswith(".7z"):
                    lz_fname = archive_fname
                    zip_fname = lz_fname[:-2] + "zip"
                    if not os.path.isfile(zip_fname):
                        cmd = "arepack -e --format=zip {}".format(lz_fname)
                        shell_stdout(cmd)
                        sys.stderr.write("Using arepack! make sure it is installed\n")
                        sys.stderr.flush()
            if len(Dibco.urls[partition]) == 2:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    gt_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][1].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_two_rar_stream(input_rar, gt_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    zip_gt_fname = root + "/" + Dibco.urls[partition][1].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        zip_input_fname = zip_input_fname[:-2] + "zip"
                        zip_gt_fname = zip_gt_fname[:-2] + "zip"
                    input_zip = zipfile.ZipFile(zip_input_fname)
                    gt_zip = zipfile.ZipFile(zip_gt_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_two_rar_stream(input_zip, gt_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
            else:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_rar_stream(input_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        zip_input_fname = zip_input_fname[:-2] + "zip"
                    input_zip = zipfile.ZipFile(zip_input_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_rar_stream(input_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
        id_data = list(data.items())
        self.sample_ids = [sample[0] for sample in id_data]
        self.inputs = [sample[1][0] for sample in id_data]
        self.gt = [sample[1][1] for sample in id_data]

    def __getitem__(self, item):
        input_img = self.input_transform(self.inputs[item])
        gt = self.gt_transform(self.gt[item])
        input_img,gt, original_img=self.crop(input_img, gt)
        return self.plasma(input_img,gt,original_img)

    def __len__(self):
        return len(self.sample_ids)

    def __add__(self, other):
        res = Dibco(partitions=[])
        res.root = self.root
        res.input_transform = self.input_transform
        res.gt_transform = self.gt_transform
        res.crop = self.crop
        res.sample_ids = self.sample_ids + other.sample_ids
        res.inputs = self.inputs + other.inputs
        res.gt = self.gt + other.gt
        return res
