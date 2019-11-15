from .util import resumable_download, mkdir_p
import torch
import torchvision
import rarfile
from PIL import Image
from StringIO import StringIO
from commands import getoutput as run_in_bash


class RandomCropTo(object):
    def __init__(self,minimum_size=[512,512],pad_if_needed=False):
        self.minimum_width, self.minimum_height = minimum_size
        self.pad_if_needed=pad_if_needed

    def __call__(self,input,gt):
        width,height,_=input.size()
        if (width<self.minimum_width or height<self.minimum_height) and self.pad_if_needed:
            if width<self.minimum_width:
                x_needed = self.minimum_width - width
            else:
                x_needed = 0

            if height < self.minimum_height:
                y_needed = self.minimum_height-height
            else:
                y_needed = 0
            input = torch.nn.functional.pad(input,y_needed/2,y_needed-y_needed/2,
                                            x_needed/2,x_needed-x_needed/2)
            gt = torch.nn.functional.pad(gt, y_needed / 2, y_needed - y_needed / 2,
                                         x_needed / 2, x_needed - x_needed / 2)
        max_left = min(self.minimum_width - width, 1)
        max_top = min(self.minimum_height - height, 1)
        left = torch.randint(low=0, high=max_left, size=[1])[0]
        top = torch.randint(low=0, high=max_top, size=[1])[0]
        right = left + self.minimum_width
        bottom = top + self.minimum_height
        return input[:,left:right,top:bottom], gt[:,left:right,top:bottom]

transform_gray = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x:torch.cat([x,1-x])
])


class Dibco:
    urls = {
        "2009_HW":["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar",
                   "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar"],
        "2009_P": [
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar",
            "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar"],
        "2010":["http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar",
                "http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar"],

        "2011_P":["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar"],
        "2011_HW": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar"],
        "2013": ["http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar"],
        "2014":["http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar",
                "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar"],
    }

    @staticmethod
    def load_single_rar_stream(rarstream):
        name2img = lambda x: Image.open(StringIO(rarstream.read(rarstream.getinfo(x)))).copy()
        id2gt = {n.split("/")[1].split("_")[0].split(".")[0]: name2img(n) for n in rarstream.namelist() if "/" in n and "_" in n.split("/")[-1] and "skelGT" not in n}
        id2in = {n.split("/")[1].split("_")[0].split(".")[0]: name2img(n) for n in rarstream.namelist() if "/" in n and "_" not in n.split("/")[-1]}
        print id2gt.keys()
        print id2in.keys()
        assert set(id2gt.keys())==set(id2in.keys())
        return {k:(id2in[k],id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def load_two_rar_stream(input_rarstream,gt_rarstream,):
        in_name2img = lambda x: Image.open(StringIO(input_rarstream.read(input_rarstream.getinfo(x)))).copy()
        gt_name2img = lambda x: Image.open(StringIO(gt_rarstream.read(gt_rarstream.getinfo(x)))).copy()
        id2in = {n.split("/")[1].split("_")[0].split(".")[0]: in_name2img(n) for n in input_rarstream.namelist() if "/" in n and "_" not in n.split("/")[-1]}
        id2gt = {n.split("/")[1].split("_")[0].split(".")[0]: gt_name2img(n) for n in gt_rarstream.namelist() if "/" in n}
        print input_rarstream.namelist()
        print gt_rarstream.namelist()
        print id2gt.keys()
        print id2in.keys()
        assert set(id2gt.keys())==set(id2in.keys())
        return {k:(id2in[k],id2gt[k]) for k in id2gt.keys()}

    def __init__(self,partitions=["2009_HW","2009_P"],crop_sz=[512,512],root="/tmp/dibco",input_transform=transform_gray,gt_transform=transform_gray):
        self.root=root
        self.input_transform=input_transform
        self.gt_transform=gt_transform
        if crop_sz is not None:
            self.crop = RandomCropTo(crop_sz)
        else:
            self.crop = lambda x, y: (x, y)
        data = {}
        for partition in partitions:
            for url in Dibco.urls[partition]:
                resumable_download(url,root)
            if len(Dibco.urls[partition])==2:
                train_rar = rarfile.RarFile(root+"/"+Dibco.urls[partition][0].split("/")[-1])
                test_rar = rarfile.RarFile(root+"/"+Dibco.urls[partition][1].split("/")[-1])
                data.update({partition+"/"+k:v for k,v in Dibco.load_two_rar_stream(train_rar,test_rar).items()})
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
