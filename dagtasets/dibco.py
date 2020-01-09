from __future__ import print_function
import sys
import math
from .util import resumable_download, mkdir_p, shell_stdout
import torch
import torchvision
import rarfile
import os
import zipfile
from PIL import Image

try:
    from StringIO import StringIO as FileReadWrapper
except:
    from io import BytesIO as FileReadWrapper


class RandomCropTo(object):
    """Functor scaling and cropping pairs of tensor images.

    """
    def __init__(self,minimum_size=[512,512], pad_if_needed=True, scale_range=(1.0, 1.0)):
        self.minimum_width, self.minimum_height = minimum_size
        self.pad_if_needed = pad_if_needed
        self.scale_range = scale_range

    def __call__(self, input_img, gt):
        print(input_img.size(),gt.size())
        width, height, _ = input_img.size()
        if self.scale_range != (1.0, 1.0):
            scale = self.scale_range[0]+float(torch.rand(1))*(self.scale_range[1] - self.scale_range[0])
            width = int(math.round(width*scale))
            height = int(math.round(height*scale))
            input_img = torch.nn.functional.interpolate(input_img.unsqueeze(dim=0), (width, height), mode="bilinear")
            gt = torch.nn.functional.interpolate(gt.unsqueeze(dim=0), (width, height), mode="nearest")
            input_img, gt = torch.squeeze(input_img, 0), torch.squeeze(gt, 0)
        if (width<self.minimum_width or height<self.minimum_height) and self.pad_if_needed:
            if width<self.minimum_width:
                x_needed = self.minimum_width - width
            else:
                x_needed = 0

            if height < self.minimum_height:
                y_needed = self.minimum_height-height
            else:
                y_needed = 0
            input_img = torch.nn.functional.pad(input_img, (int(y_needed / 2), int(y_needed - y_needed / 2),
                                                            int(x_needed/2), int(x_needed-x_needed/2)))
            gt = torch.nn.functional.pad(gt, (int(y_needed / 2), int(y_needed - y_needed / 2),
                                         int(x_needed / 2), int(x_needed - x_needed / 2)))
        max_left = max(width- self.minimum_width , 1)
        max_top = max(height- self.minimum_height, 1)
        left = torch.randint(low=0, high=max_left, size=[1])[0]
        top = torch.randint(low=0, high=max_top, size=[1])[0]
        right = left + self.minimum_width
        bottom = top + self.minimum_height
        input_img,gt = input_img[:, left:right, top:bottom], gt[:, left:right, top:bottom]
        return input_img, gt

transform_gray_train = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x:torch.cat([x,1-x])
])

transform_color_train = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
    torchvision.transforms.ToTensor()
])

transform_gray_inference = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x:torch.cat([x,1-x])
])


class Dibco:
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

        "2014":["http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar",
                "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar"],
        "2016":["https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
                "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip"],
        "2017":["https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z",
                "https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z"],
        "2018":["http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018_Dataset.zip",
                "http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018-GT.zip"]
    }

    @staticmethod
    def load_single_rar_stream(compressed_stream):
        def name2img(x):
            return Image.open(FileReadWrapper((compressed_stream.read(compressed_stream.getinfo(x))))).copy()
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: name2img(n) for n in compressed_stream.namelist() if "." in n and "_" in n.split("/")[-1] and "skelGT" not in n}
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: name2img(n) for n in compressed_stream.namelist() if "." in n and "_" not in n.split("/")[-1]}
        assert set(id2gt.keys())==set(id2in.keys())
        return {k:(id2in[k],id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def load_two_rar_stream(input_compressed_stream, gt_compressed_stream, ):
        def in_name2img(x):
            return Image.open(FileReadWrapper((input_compressed_stream.read(input_compressed_stream.getinfo(x))))).copy()
        def gt_name2img(x):
            return Image.open(FileReadWrapper((gt_compressed_stream.read(gt_compressed_stream.getinfo(x))))).copy()
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: in_name2img(n) for n in input_compressed_stream.namelist() if "." in n and "_" not in n.split("/")[-1]}
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: gt_name2img(n) for n in gt_compressed_stream.namelist() if "." in n and "skelGT" not in n and not n.endswith(".dat")}
        assert set(id2gt.keys()) == set(id2in.keys())

        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def Dibco2009(crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray_train,gt_transform=transform_gray_train):
        return Dibco(partitions=["2009_HW","2009_P"],crop_sz=crop_sz,root=root,input_transform=input_transform,gt_transform=gt_transform)

    @staticmethod
    def Dibco2010(crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray_train,gt_transform=transform_gray_train):
        return Dibco(partitions=["2010"],crop_sz=crop_sz,root=root,input_transform=input_transform,gt_transform=gt_transform)

    @staticmethod
    def Dibco2011(crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray_train,gt_transform=transform_gray_train):
        return Dibco(partitions=["2011_P","2011_HW"],crop_sz=crop_sz,root=root,input_transform=input_transform,gt_transform=gt_transform)

    @staticmethod
    def Dibco2013(crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray_train,gt_transform=transform_gray_train):
        return Dibco(partitions=["2013"],crop_sz=crop_sz,root=root,input_transform=input_transform,gt_transform=gt_transform)

    def Dibco2014(crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray_train,gt_transform=transform_gray_train):
        return Dibco(partitions=["2014"],crop_sz=crop_sz,root=root,input_transform=input_transform,gt_transform=gt_transform)


    def __init__(self,partitions=["2009_HW","2009_P"],crop_sz=[512,512],root="/tmp/dibco",train=True,scale_range=None):
        if train:
            self.input_transform = transform_gray_train
            self.gt_transform = transform_gray_train
        else:
            self.input_transform = transform_gray_inference
            self.gt_transform = transform_gray_inference
        self.train=train
        self.root=root
        if crop_sz is not None or scale_range != None:
            if scale_range is None:
                scale_range = (1.0,1.0)
            self.crop = RandomCropTo(crop_sz, scale_range=scale_range)
        else:
            self.crop = lambda x, y: (x, y)
        data = {}
        for partition in partitions:
            for url in Dibco.urls[partition]:
                resumable_download(url,root)
                if url.endswith(".7z"):
                    lz_fname=root+"/"+url.split("/")[-1]
                    zip_fname=lz_fname[:-2]+"zip"
                    if not os.path.isfile(zip_fname):
                        cmd="arepack -e --format=zip {}".format(lz_fname)
                        shell_stdout(cmd)
                        os.stderr.write("Using arepack! make sure it is installed\n")
                        os.stderr.flush()
            if len(Dibco.urls[partition])==2:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root+"/"+Dibco.urls[partition][0].split("/")[-1])
                    gt_rar = rarfile.RarFile(root+"/"+Dibco.urls[partition][1].split("/")[-1])
                    samples={partition+"/"+k:v for k,v in Dibco.load_two_rar_stream(input_rar,gt_rar).items()}
                    data.update(samples)
                    print("Dibco2({}):{}".format(repr(Dibco.urls[partition]), len(samples)))
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname=root+"/"+Dibco.urls[partition][0].split("/")[-1]
                    zip_gt_fname = root + "/" + Dibco.urls[partition][1].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        zip_input_fname=zip_input_fname[:-2]+"zip"
                        zip_gt_fname = zip_gt_fname[:-2] + "zip"
                    input_zip = zipfile.ZipFile(zip_input_fname)
                    gt_zip = zipfile.ZipFile(zip_gt_fname)
                    samples={partition+"/"+k:v for k,v in Dibco.load_two_rar_stream(input_zip,gt_zip).items()}
                    data.update(samples)
                    print("Dibco2({}):{}".format(repr(Dibco.urls[partition]), len(samples)))
                else:
                    raise ValueError("Unknown file type")
            else:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    samples={partition + "/" + k: v for k, v in Dibco.load_single_rar_stream(input_rar).items()}
                    data.update(samples)
                    print("Dibco1({}):{}".format(repr(Dibco.urls[partition]),len(samples)))
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        zip_input_fname=zip_input_fname[:-2]+"zip"
                    input_zip = zipfile.ZipFile(zip_input_fname)
                    samples={partition+"/"+k: v for k, v in Dibco.load_single_rar_stream(input_zip).items()}
                    data.update(samples)
                    print("Dibco1({}):{}".format(repr(Dibco.urls[partition]), len(samples)))
                else:
                    raise ValueError("Unknown file type")
        id_data=list(data.items())
        self.sample_ids=[sample[0] for sample in id_data]
        self.inputs = [sample[1][0] for sample in id_data]
        self.gt = [sample[1][1] for sample in id_data]


    def __getitem__(self, item):
        input = self.input_transform(self.inputs[item])
        gt = self.gt_transform(self.gt[item])
        return self.crop(input, gt)

    def __len__(self):
        return len(self.sample_ids)

    def __add__(self,other):
        res= Dibco(partitions=[])
        res.root=self.root
        res.input_transform=self.input_transform
        res.gt_transform=self.gt_transform
        res.crop=self.crop
        res.sample_ids=self.sample_ids+other.sample_ids
        res.inputs = self.inputs + other.inputs
        res.gt = self.gt + other.gt
        return res