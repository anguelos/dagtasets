import os
import torch.utils.data as data
from skimage import io
import torchvision.transforms as transforms

input_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class DagDataset(data.Dataset):
    def __init__(self,root_file,root_dir=None,in_ram=False,input_transform=input_transform,target_transform=None):
        self.root_file = root_file
        if root_dir is None:
            root_dir = os.path.dirname(root_file)
        self.root_dir = root_dir
        self.in_ram = in_ram

        # Making transform always applicable
        if input_transform is None:
            self.input_transform = lambda x:x
        else:
            self.input_transform = input_transform
        if target_transform is None:
            self.target_transform = lambda x:x
        else:
            self.target_transform = target_transform

        self.items=[]
        for line in open(root_file).read().strip().split("\n"):
            fname=line[:line.find("\t")]
            caption = eval(line[line.find("\t")+1:])
            if in_ram:
                img_path = os.path.join(self.root_dir,fname)
                input = io.imread(img_path)
            else:
                input = img_path
            self.items.append((input,caption))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        if self.in_ram:
            image , caption = self.items[item]
        else:
            filename, caption = self.items[item]
            image = im.imread(filename)
        input = self.input_transform(image)
        output = self.target_transform(caption)
        return input, output