from unrealcv import client
import numpy as np
from matplotlib import pyplot as plt

from dagtasets import mkdir_p,load_image_float,save_image_float

from .util import *

def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    import StringIO, PIL.Image
    img = PIL.Image.open(StringIO.StringIO(res))
    return np.asarray(img)

def read_npy(res):
    import StringIO
    return np.load(StringIO.StringIO(res))

def alpha_blend(bg_img,mask_img,fg,top_left=(0,0)):
    #https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    top,left=top_left
    bottom=mask_img.shape[0]+top
    right = mask_img.shape[1] + left
    assert bottom<=bg_img.shape[0] and right<=bg_img.shape[1]
    res_img=bg_img.copy()
    mask_img=mask_img.reshape(mask_img.shape+(1,))
    if len(fg.shape)==1:
        fg=fg.reshape([1,1,3])

    res_img[top:bottom, left:right, :3]=mask_img*fg+(1-mask_img)*bg_img[top:bottom,left:right,:3]
    return res_img


class ExternalSceneRenderer():
    def __init__(self):
        client.connect()
        if not client.isconnected():
            raise IOError

    def __del__(self):
        client.disconnect()

    def _get_status_str(self):
        return client.request('vget /unrealcv/status')

    def _get_current_img(self):
        png_str = client.request('vget /camera/0/lit png')
        image = read_png(png_str)
        return image/255.0

    def _get_current_normal(self):
        return (read_npy(client.request('vget /camera/0/normal npy'))/255.0)-.5

    def _get_current_objmask(self):
        return read_npy(client.request('vget /camera/0/object_mask npy'))


    def get_paste_regions(self):
        normal_e=.07
        normal=self._get_current_normal()
        same_normal=np.zeros(normal.shape[:2])
        same_normal[:-1,:-1]=(np.abs(normal[1:,1:,:]-normal[:-1,:-1,:])<normal_e).prod(axis=2)
        print 'range ',normal[:,:,2].min(),normal[:,:,2].max()
        horizontal_surface=np.abs(normal[:,:,2])<normal_e
        #plt.imshow(same_normal,cmap='gray');plt.imshow(normal[:,:,2],cmap='gray');plt.show()
        plt.imshow(horizontal_surface*same_normal, cmap='gray');
        plt.show()

    def paste_image(self,text_caption,fg_color=np.array([1.0,.0,1.])):
        text_image,_=render_text(caption=text_caption,image_width=200,image_height=150)
        from matplotlib import pyplot as plt
        plt.imshow(text_image,cmap='gray');plt.show()
        bg=self._get_current_img()/255.0
        return alpha_blend(bg,1-text_image,fg_color)
