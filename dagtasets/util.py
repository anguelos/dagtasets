from commands import getstatusoutput,getoutput as shell_stdout
import os
import errno
import cv2
import tarfile
import torch
import torch.nn.functional as F

def extract(archive,root=None):
    if archive.endswith(".tar.gz"):
        if root is None:
            cmd="tar -xpvzf {}".format(archive)
        else:
            cmd = 'mkdir -p {};tar -xpvzf {} -C{}'.format(root,archive,root)
        status,output = getstatusoutput(cmd)
        if status != 0 :
            raise Exception("cmd '{}' failed with exit code {}.".format(cmd,status))
    else:
        raise NotImplementedError()


def check_os_dependencies():
    program_list=["wget"]
    return all([shell_stdout("which "+prog) for prog in program_list])


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def resumable_download(url,save_dir):
    mkdir_p(save_dir)
    download_cmd = 'wget --directory-prefix=%s -c %s' % (save_dir, url)
    print "Downloading %s ... "%url,
    shell_stdout(download_cmd)
    print "done"
    return os.path.join(save_dir,url.split("/")[-1])


def load_image_float(fname,as_color=True):
    if as_color:
        img=cv2.cvtColor(cv2.imread(fname,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    else:
        img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
    return img/255.0


def save_image_float(img,fname,as_color=True,mkdir=True):
    img=(img*255).astype('uint8')
    if as_color:
        if len(img.shape)==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    else:
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(fname,img)


class RandomPadAndNormalise(object):
    """Transforms a 3D tensor by padding with noise and standarizing the tensor.

    The functor is parametrised by a tuple containing the desired width and
    height.
    The input is assumed to be a 3D tensor containing a sample with dimensions
    [channels x height x width]. The output is a tensor padded or cropped along
    width and height to match the desired image size. The padding pixels follows
    the same distribution as the image pixels across all channels.

    :param width_height: A tuple with two integers containing the desired width
        and height of the output tensor.
    :param standarise: If True, the tensor pixels will be standarised to a zero-
        mean and unit deviation.
    :param e: A close to 0 value added to the standarisation divider.
    """

    def __init__(self, width_height, standarise=True, e=1e-20):
        self.e = e
        self.desired_height_width = width_height
        self.standarise = standarise

    def __call__(self, tensor):
        channels, height, width = tensor.shape
        mean = float(tensor.view(-1).mean())
        std = float(tensor.view(-1).std())
        # TODO: (anguelos) use mean and variance by channel.
        # mean=tensor.view([channels,height*width]).mean(dim=1)
        # std = tensor.view([channels, height * width]).std(dim=1)
        if height < self.desired_height_width[0]:
            rnd_pads = torch.empty(
                [channels, self.desired_height_width[0] - height,
                 width]).normal_(mean, std)
            tensor = torch.cat((tensor, rnd_pads), 1)
        if width < self.desired_height_width[1]:
            rnd_pads = torch.empty([channels, self.desired_height_width[0],
                                    self.desired_height_width[
                                        1] - width]).normal_(mean, std)
            tensor = torch.cat((tensor, rnd_pads), 2)
        if self.standarise:
            tensor = (tensor - mean) / (std + self.e)
        # Performing cropping if need be.
        tensor = tensor[:, :self.desired_height_width[0],
                 :self.desired_height_width[1]]
        return tensor


class SequenceCollateFunctor(object):
    def __init__(self,pad_captions,sort_by):
        """Colation Functor for sequence DataLoader.

        :param pad_captions: If True, captions will be a 2D Tensor padded with zeros.
            If False, they are concatenated into a single vector as libcudnn expects them.
        :param sort_by: One of ["none", "width", "height", "transcription"] if sequences
            are to be packed, libcudnn expects the samples in decreassing order.
        """
        assert sort_by .lower() in ["none", "width", "height", "transcription"]
        self.pad_captions=pad_captions
        self.sort_by = sort_by
    def __call__(self,batch_list):
        """Colate Functor for sequence DataLoader.

        :param batch_list: A list of tuples with a 3D tensor representing an image and
            1D integer tensor representing the caption.
        :return: A tuple with the 4D BCHW tensor, a 1D Tensor  with the widths, a 1D Tensor
            with the heights,  a 1D or 2D Tensor  with the captions, and  a 1D Tensor
            with the caption_lengths.
        """
        if self.sort_by=="width":
            batch_list.sort(key=lambda x: x[0].size()[2], reverse=True)
        elif self.sort_by=="height":
            batch_list.sort(key=lambda x: x[0].size()[1], reverse=True)
        elif self.sort_by=="transcription":
            batch_list.sort(key=lambda x: len(x[1]), reverse=True)
        captions, caption_lengths, img_list, heights, widths = zip(
            *[(item[1], item[1].size(0), item[0], item[0].size(1), item[0].size(2)) for item in batch_list])
        batch_width = max(widths)
        batch_height = max(heights)
        max_caption_length = max(caption_lengths)
        pad_img = lambda x: F.pad(x, (0, batch_width - x.size(2), 0, batch_height - x.size(1), 0, 0))
        batch = torch.stack([pad_img(img) for img in img_list])
        if self.pad_captions:
            pad_caption = lambda x: F.pad(x, (0, max_caption_length - x.size(0)))
            captions = torch.stack([pad_caption(caption) for caption in captions])
        else:
            captions = torch.cat(captions)
        return batch, torch.LongTensor(widths), torch.LongTensor(heights), captions, torch.LongTensor(caption_lengths)


def get_variable_lenght_dataloader(dataset,batch_size=10,shuffle=True,num_workers=4,pad_captions=True,sort_by="width"):
    collator=SequenceCollateFunctor(pad_captions=pad_captions,sort_by=sort_by)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collator)
    return data_loader
