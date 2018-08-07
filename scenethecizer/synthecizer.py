    # -*- coding: utf-8 -*-
import cairo
import codecs
import pangocairo
import string
from commands import getoutput as go

import cv2
import numpy as np
import glob
import pango
import scipy.interpolate as interpolate
from matplotlib import pyplot as plt

import lm_util
import random
import skimage
import skimage.filters


import os

from dagtasets import mkdir_p,load_image_float,save_image_float

from .util import *



class PixelOperation(object):
    def __init__(self):
        self.parameters=self.generate_parameters()

    def generate_parameters(self):
        raise NotImplementedError

    def deterministic(self, img, parameter_list):
        raise NotImplemented

    def __call__(self, img, update_state=True):
        res=self.deterministic(img, self.parameters)
        if update_state:
            self.parameters=self.generate_parameters()
        return res

    def apply_on_image(self, img, ltrb, update_state=True):
        return self(img, update_state   ), ltrb



class DocumentNoise(PixelOperation):
    def __init__(self,img_shape,noise_sparsity=500):
        self.img_shaep=img_shape
        self.nb_pixels=img_shape[0] * img_shape[1]
        self.noise_sparsity=noise_sparsity
        super(DocumentNoise,self).__init__()

    def generate_parameters(self):
        low_noise_indexes = np.random.randint(0, self.nb_pixels * 255,
                                              self.nb_pixels / self.noise_sparsity)
        return [low_noise_indexes, .8, [1, 1], [1, 1], 0.6]

    def deterministic(self, img, parameter_list=[np.array([0], dtype='int32'),.5, [1, 1], [1, 1],0.6]):
        low_noise_indexes, low_pass_ignore, low_pass_sigma, high_pass_sigma, low_pass_range = parameter_list
        #img = img[:, :] / 255.0
        low_noise = np.zeros(img.shape[:2], dtype='float')
        low_noise.reshape(-1)[low_noise_indexes / 255] = low_noise_indexes % \
                                                         255
        # low_noise=(np.random.rand(img.shape[0],img.shape[1])>(1.0/noise_sparsity)).astype('float32')
        low_pass = skimage.filters.gaussian(low_noise[:, :], low_pass_sigma,multichannel=False)

        low_pass = (low_pass - np.min(low_pass)) / (
                np.max(low_pass) - np.min(low_pass))

        low_pass -= low_pass_ignore
        low_pass = low_pass * (low_pass > 0)
        high_pass = img * ((np.random.rand(img.shape[0], img.shape[1]) * (
                1 - low_pass_range) + (low_pass_range)))[:,:,None]
        high_pass = skimage.filters.gaussian(high_pass[:, :,:], high_pass_sigma,multichannel=True)
        # return 1-high_pass
        res = (low_pass[:,:,None] + high_pass)  # np.maximum(low_pass,img)
        res = res - res.min() / (res.max() - res.min())
        return res

class ImageBackground(PixelOperation):
    def __init__(self,image_fname_list=None,blend='mean',bg_resize='tile'):
        if image_fname_list is None or image_fname_list == []:
            self.image_list=[np.random.rand(100,100,3) for _ in range(2)]
        else:
            self.image_list=[]
            for img_fname in image_fname_list:
                bg_img=load_image_float(img_fname,as_color=True)
                self.image_list.append(bg_img)
        if blend=="max":
            self.blend=self.blend_max
        elif blend=="min":
            self.blend = self.blend_min
        elif blend=="mean":
            self.blend = self.blend_mean
        else:
            raise ValueError
        if bg_resize=="tile":
            self.resize_bg = self.resize_bg_tile
        elif bg_resize=="scale":
            self.resize_bg = self.resize_bg_scale
        else:
            raise ValueError
        super(ImageBackground, self).__init__()  # The list must exist before parent constructor

    def generate_parameters(self):
        return [random.choice(self.image_list)]

    def deterministic(self, img, parameter_list):
        bg, = parameter_list
        bg = self.resize_bg(bg,img.shape)
        if len(bg.shape)!=len(img.shape):
            if len(bg.shape)==2 and len(img.shape)==3:
                bg=np.array([bg,bg,bg]).swapaxes(2,0)
            elif len(bg.shape)==3 and len(img.shape)==2:
                img=np.array([img,img,img]).swapaxes(1,0).swapaxes(1,2)
            else:
                raise ValueError
        res = self.blend(img,bg)
        return res

    def __call__(self,img, update_state=True):
        res=self.deterministic(img,self.parameters)
        if update_state:
            self.parameters=self.generate_parameters()
        return res

    def resize_bg_tile(self,bg,fg_shape):
        fg_width = fg_shape[1]
        fg_height = fg_shape[0]
        bg_width = bg.shape[1]
        bg_height = bg.shape[0]
        res=np.empty([fg_height, fg_width, 3])
        for left in range(0,fg_width,bg_width):
            right=min(left+bg_width,fg_width)
            width=right-left
            for top in range(0,fg_height,bg_height):
                bottom=min(top+bg_height,fg_height)
                height=bottom-top
                res[top:bottom,left:right,:]=bg[:height,:width,:]
        return res

    def resize_bg_scale(self,bg,fg_shape):
        return cv2.resize(bg,[fg_shape[1],fg_shape[0]]) # TODO(anguelos) make cv2 optional

    def blend_max(self,fg,bg):
        return np.maximum(fg,bg)

    def blend_min(self,fg,bg):
        return np.minimum(fg,bg)

    def blend_mean(self,fg,bg):
        return (fg+bg)/2


class GeometricOperation(object):
    def __init__(self):
        self.parameters=self.generate_parameters()

    def generate_parameters(self):
        raise NotImplementedError

    def deterministic(self, X, Y, parameter_list):
        raise NotImplemented

    def __call__(self, x_coordinate_list, y_coordinate_list, update_state=True):
        res_x_coordinate_list = []
        res_y_coordinate_list = []
        #params=self.generate_parameters()
        for k in range(len(x_coordinate_list)):
            X = x_coordinate_list[k].copy()
            Y = y_coordinate_list[k].copy()
            X, Y = self.deterministic(X, Y, self.parameters)
            res_x_coordinate_list.append(X)
            res_y_coordinate_list.append(Y)
        if update_state:
            self.parameters=self.generate_parameters()
        return res_x_coordinate_list, res_y_coordinate_list

    def apply_on_image(self, img, ltrb, update_state=True):
        l = ltrb[:, 0]
        t = ltrb[:, 1]
        r = ltrb[:, 2]
        b = ltrb[:, 3]
        X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        in_x_coords = [X, l, r]#[X, l, l, r, r]
        in_y_coords = [Y, t, b]#[Y, t, b, t, b]
        out_x_coords, out_y_coords = self(in_x_coords,in_y_coords, update_state)
        #[X, x1, x2, x3, x4] = out_x_coords
        [X, x1, x2] = out_x_coords
        #[Y, y1, y2, y3, y4] = out_y_coords
        [Y, y1, y2] = out_y_coords
        res_img = raw_interp2(X, Y, img)
        res_ltrp = np.empty(ltrb.shape)
        res_ltrp[:, 0] = x1#np.min([x1,x2],axis=0)#,x3,x4], axis=0)
        res_ltrp[:, 2] = x2##np.max([x1, x2],axis=0)#, x3, x4], axis=0)
        res_ltrp[:, 1] = y1#np.min([y1, y2],axis=0)#, y3, y4], axis=0)
        res_ltrp[:, 3] = y2#np.max([y1, y2],axis=0)#, y3, y4], axis=0)
        return res_img, res_ltrp


class GeometricSequence(GeometricOperation):
    def generate_parameters(self):
        return None

    def __init__(self,*args):
        super(GeometricSequence,self).__init__()
        self.transform_sequences=args

    def deterministic(self, X, Y,parameters):
        del parameters
        for transform in self.transform_sequences:
            X, Y = transform(X, Y)
        return X,Y


class GeometricCliper(GeometricOperation):
    def __init__(self,clip_ltrb):
        self.clip_ltrb=clip_ltrb
        super(GeometricCliper, self).__init__()

    def generate_parameters(self):
        return self.clip_ltrb

    def deterministic(self, X, Y, parameter_list):
        x_min, y_min, x_max, y_max = parameter_list
        X[X < x_min] = x_min
        X[X > x_max] = x_max
        Y[Y < y_min] = y_min
        Y[Y > y_max] = y_max
        return X, Y

class GeometricTextlineWarper(GeometricOperation):
    def __init__(self,page_size, letter_height, num_points=8):
        self.num_points=num_points
        self.page_size=page_size
        self.letter_height=letter_height
        super(GeometricTextlineWarper, self).__init__()

    def generate_parameters(self):
        xpoints = (np.random.rand(self.num_points) * np.array(
            [0] + [1] * (self.num_points - 1))).cumsum()
        xpoints = xpoints / xpoints.max()
        ypoints = (np.random.standard_normal(self.num_points) ) * np.array([0] + [1] * (self.num_points - 2) + [0])
        print "GENERATING",xpoints,ypoints
        return xpoints, ypoints

    def deterministic(self, X, Y, parameter_list):
        print "APPLIED",parameter_list
        xpoints, ypoints = parameter_list
        ticks = interpolate.splrep(xpoints, ypoints)
        all_x = np.linspace(0, 1, self.page_size[1])
        all_y = (interpolate.splev(all_x, ticks))
        if len(Y.shape)==2:
            Y = Y + all_y[None,:]
        elif len(Y.shape)==1:
            Y[:]=Y[:] + all_y[X[:]]
        else:
            raise ValueError
        return X, Y

class GeometricRandomTranslator(GeometricOperation):
    def __init__(self,x_sigma,x_mean,y_sigma,y_mean):
        self.x_sigma=x_sigma
        self.x_mean=x_mean
        self.y_sigma=y_sigma
        self.y_mean=y_mean
        super(GeometricRandomTranslator, self).__init__()

    def generate_parameters(self,point_shape):
        res_x = np.random.standard_normal(point_shape)*self.x_sigma + self.x_mean
        res_y = np.random.standard_normal(point_shape) * self.y_sigma + self.y_mean
        return res_x,res_y

    def deterministic(self, X, Y, parameter_list):
        return X+parameter_list[0],Y+parameter_list[1]

    def __call__(self, x_coordinate_list, y_coordinate_list, update_state=True):
        if update_state == False:
            raise NotImplementedError()
        res_x_coordinate_list = []
        res_y_coordinate_list = []
        for k in range(len(x_coordinate_list)):
            X = x_coordinate_list[k].copy()
            Y = y_coordinate_list[k].copy()
            X, Y = self.deterministic(X, Y, self.generate_parameters(X.shape))
            res_x_coordinate_list.append(X)
            res_y_coordinate_list.append(Y)
        return res_x_coordinate_list, res_y_coordinate_list

    def apply_on_image(self, img, ltrb, update_state=True):
        l = ltrb[:, 0]
        t = ltrb[:, 1]
        r = ltrb[:, 2]
        b = ltrb[:, 3]
        in_x_coords = [l,r]
        in_y_coords = [t, b]
        out_x_coords, out_y_coords = self(in_x_coords,in_y_coords, update_state=update_state)
        res_ltrb = np.empty(ltrb.shape)
        res_ltrb[:, 0] = out_x_coords[0]
        res_ltrb[:, 2] = out_x_coords[1]
        res_ltrb[:, 1] = out_y_coords[0]
        res_ltrb[:, 3] = out_y_coords[1]
        return img, res_ltrb


class GeometricBBoxDilator(GeometricOperation):
    def __init__(self,x_y_dilations,crop_to_shape=None):
        if isinstance(x_y_dilations,numbers.Number):
            self.left_pad=x_y_dilations
            self.right_pad=x_y_dilations
            self.top_pad=x_y_dilations
            self.bottom_pad=x_y_dilations
        elif len(x_y_dilations) == 2 and isinstance(x_y_dilations[0],numbers.Number):
            self.left_pad=x_y_dilations[0]
            self.right_pad=x_y_dilations[0]
            self.top_pad=x_y_dilations[1]
            self.bottom_pad=x_y_dilations[1]
        else:
            self.left_pad=x_y_dilations[0][0]
            self.right_pad=x_y_dilations[0][1]
            self.top_pad=x_y_dilations[1][0]
            self.bottom_pad=x_y_dilations[1][1]
        self.crop_to_size=crop_to_size
        super(GeometricBBoxDilator, self).__init__()

    def generate_parameters(self):
        raise NotImplementedError

    def deterministic(self, X, Y, parameter_list):
        raise NotImplementedError

    def __call__(self, x_coordinate_list, y_coordinate_list, update_state=True):
        raise NotImplementedError

    def apply_on_image(self, img, ltrb, update_state=True):
        res_ltrb=ltrb+np.array([self.left_pad,self.top_pad,self.right_pad,self.bottom_pad])[None,:]
        if self.crop_to_size:
            left=res_ltrb[:,0]
            right = res_ltrb[:, 2]
            top=res_ltrb[:,1]
            bottom = res_ltrb[:, 3]
            left[left<0]= 0
            left[left>=crop_to_shape[1]]=crop_to_shape[1]-1
            right[right < 0] = 0;
            right[right >= crop_to_shape[1]] = crop_to_shape[1] - 1

            top[top<0]= 0
            top[top>=crop_to_shape[0]]=crop_to_shape[0]-1
            bottom[bottom < 0] = 0;
            bottom[bottom >= crop_to_shape[0]] = crop_to_shape[0] - 1
        return img, res_ltrb



class Synthesizer(object):
    @staticmethod
    def get_system_fonts(only_home=True):
        if only_home:
            # TODO (anguelos) FIX THIS
            font_list = [f.split("/")[-1].split(".")[0] for f in
                         go('echo "$HOME/.fonts/"*.ttf').split()]
            font_list = [f.get_name() for f in
                         pangocairo.cairo_font_map_get_default().list_families()]
            return font_list
        else:
            raise NotImplementedError()
    alignmets = [pango.ALIGN_LEFT, pango.ALIGN_RIGHT,
                 pango.ALIGN_CENTER]  # TODO(anguelos) fix textline stichers as they rely on alignment

    def     render_page(self):
        return render_text(caption=self.current_page_caption,
                            fontname=self.current_font_name,
                            font_height=self.letter_height,
                            image_width=self.page_width,
                            image_height=self.page_height,
                            crop_edge_ltrb=self.crop_edge_ltrb,
                            alignment=self.current_alignment,
                           crop_for_height=True
                           )


    def __init__(self):  # , caption_fname=None):
        pass

    @classmethod
    def create_printed_synthecizer(cls, caption_reader, font_names=None, quantum="textlines", letter_height=30,bg_img_list=[]):
        """Synthecizer constructor.

        """

        synth = cls()
        if quantum == "textlines":
            synth.split_substrings = synth.split_textlines
        elif quantum == "words":
            synth.split_substrings = synth.split_words
        elif quantum == "letters":
            synth.split_substrings = synth.split_letters
        else:
            raise ValueError()
        synth.aligment_probs = np.array([1.0, 0.0, 0.0])
        synth.font_names = Synthesizer.get_system_fonts()
        assert len(synth.font_names) > 0
        synth.font_probs = np.ones(len(synth.font_names))
        synth.letter_height = letter_height
        synth.page_height = letter_height * 30
        synth.page_width = letter_height * 50
        synth.crop_edge_ltrb = np.array([letter_height * 5 ]*4)
        image_size=[synth.page_height,synth.page_width]
        available_fonts=Synthesizer.get_system_fonts()
        if font_names is None:
            font_list=list(available_fonts)
        else:
            font_list = list(
                set(font_names).intersection(set(available_fonts)))
        assert len(font_list)

        def font_generator():
            fnt_name = ' '
            while ' ' in fnt_name:
                fnt_name = font_list[np.random.randint(0, len(font_list))]
            return fnt_name

        synth.font_generator = font_generator
        synth.alignment_generator = lambda: pango.ALIGN_LEFT
        synth.caption_generator = caption_reader
        synth.distort = synth.distort_printed_synth
        synth.distort_operations = [ImageBackground(bg_img_list), DocumentNoise([synth.page_height,synth.page_width]),
                                        GeometricTextlineWarper(page_size=image_size, letter_height=letter_height, num_points=8),
                                        GeometricCliper(
                                            (0, 0, synth.page_width, synth.page_height))]
        synth.generate_new_page()
        return synth

    @classmethod
    def create_handwriten_synthecizer(cls, caption_reader, font_names=("Pacifico","Cookie","Gaegu","Sacramento","Tangerine","Allura"), quantum="textlines", letter_height=30,bg_img_list=[]):
        """Synthecizer constructor.

        ("Byron", "ALincolnFont", "Caligraf 1435", "Celine Dion Handwriting")

        """
        synth = cls()
        if quantum == "textlines":
            synth.split_substrings = synth.split_textlines
        elif quantum == "words":
            synth.split_substrings = synth.split_words
        elif quantum == "letters":
            synth.split_substrings = synth.split_letters
        else:
            raise ValueError()
        synth.aligment_probs = np.array([1.0, 0.0, 0.0])
        synth.font_names = Synthesizer.get_system_fonts()
        assert len(synth.font_names) > 0
        synth.font_probs = np.ones(len(synth.font_names))

        synth.letter_height = letter_height
        synth.page_height = letter_height * 50
        synth.page_width = letter_height * 40
        synth.crop_edge_ltrb = np.array([letter_height * 5 ]*4)

        image_size=[synth.page_height,synth.page_width]

        available_fonts=Synthesizer.get_system_fonts()
        font_list = list(
            set(font_names).intersection(set(available_fonts)))

        assert len(font_list)

        def font_generator():
            fnt_name = ' '
            while ' ' in fnt_name:
                fnt_name = font_list[np.random.randint(0, len(font_list))]
            return fnt_name

        synth.font_generator = font_generator

        synth.alignment_generator = lambda: pango.ALIGN_LEFT

        synth.caption_generator = caption_reader

        synth.distort = synth.distort_handwriten_synth
        #synth.distort_operations=[ImageBackground(),DocumentNoise(),GeometricSequence(GeometricTextlineWarper(image_size),GeometricCliper((0,0,image_width,image_height)))]
        synth.distort_operations = [ImageBackground(bg_img_list), DocumentNoise([synth.page_height,synth.page_width]),
                                        GeometricTextlineWarper(page_size=image_size, letter_height=letter_height, num_points=8),
                                        GeometricCliper(
                                            (0, 0, synth.page_width, synth.page_height))]
        synth.generate_new_page()
        return synth

    def generate_new_page(self):

        self.current_font_name = self.font_generator()
        self.current_alignment = self.alignment_generator()

        nb_chars = (self.page_width / self.letter_height) * (
                    self.page_height / self.letter_height)

        self.current_page_caption = self.caption_generator.read_str(nb_chars)

        self.current_img, self.current_graphene_ltrb = self.render_page()

        inside_page = self.current_graphene_ltrb[:, 1] < (
                    self.page_height + self.crop_edge_ltrb[1])
        self.current_graphene_ltrb = self.current_graphene_ltrb[inside_page, :]

        self.current_page_caption = u''.join((np.array(
            list(self.current_page_caption), dtype="unicode")[
            inside_page]).tolist())

        self.distort()  # anguelos: maybe this should go elsewhere

    def distort(self):
        """Takes a new page and applies all defined distortions. Not a real method, assigned dynamically by classmethod
        constructor."""
        # Flat inheritance
        raise NotImplementedError()

    def distort_printed_synth(self):
        """Takes a new page and applies all distortions defined for printed.

        :return:
        """
        img, bboxes = self.current_img, self.current_graphene_ltrb
        gt_img = img.copy()
        for operation in self.distort_operations:
            if isinstance(operation,GeometricOperation):
                print "BEFORE GT:"
                gt_img,_ = operation.apply_on_image(gt_img, bboxes,update_state=False)
            img, bboxes = operation.apply_on_image(img, bboxes)
            print "AFTER IMG\n\n\n\n"

        self.current_graphene_ltrb = bboxes.copy()
        self.current_img = img.copy()
        self.gt_img = gt_img

        ranges = self.split_substrings(self.current_page_caption, bboxes)
        roi_captions, bboxes = self.stich_ranges(self.current_page_caption,
                                                 ranges, bboxes)
        #tr_bboxes = self.distort_translate_bboxes_defaults(bboxes, img.shape[1],
        #                                                   img.shape[0])
        self.current_roi_captions = roi_captions
        self.current_roi_ltrb = bboxes


    def distort_handwriten_synth(self):
        """Takes a new pag  e and applies all defined handwriting distortions

        :return:
        """
        img, bboxes = self.current_img, self.current_graphene_ltrb
        gt_img=img.copy()
        for operation in self.distort_operations:
            if isinstance(operation,GeometricOperation):
                gt_img,_ = operation.apply_on_image(gt_img, bboxes,update_state=False)
            img, bboxes = operation.apply_on_image(img, bboxes)

        self.current_graphene_ltrb = bboxes.copy()
        self.current_img = img.copy()
        self.gt_img = gt_img
        self.final_img=img.copy()

        ranges = self.split_substrings(self.current_page_caption, bboxes)
        roi_captions, bboxes = self.stich_ranges(self.current_page_caption,
                                                 ranges, bboxes)
        #tr_bboxes = self.distort_translate_bboxes_defaults(bboxes, img.shape[1],
        #                                                   img.shape[0])
        self.current_roi_captions = roi_captions
        self.current_roi_ltrb = bboxes

    def plot_current_page(self, pause=True):
        """Auxiliary method for ploting interactivelly the page.

        :param pause:
        :return:
        """
        plt.plot(self.current_roi_ltrb[:, [0, 0, 2, 2, 0]].T,
                 self.current_roi_ltrb[:, [1, 3, 3, 1, 1]].T)
        plt.imshow(self.current_img, cmap='gray', vmin=0.0, vmax=1.0);
        plt.ylim((self.current_img.shape[0], 0));
        plt.xlim((0, self.current_img.shape[1]));
        plt.plot(self.current_roi_ltrb[:, [0, 0, 2, 2, 0]].T,
                 self.current_roi_ltrb[:, [1, 3, 3, 1, 1]].T)
        if pause:
            plt.show()

    def get_visible_idx(self, caption):
        """Given a string, returns a boolean numpy array with True for visible characters and False for whitespace.

        :param caption:
        :return:
        """
        return np.array([c not in string.whitespace for c in caption],
                        dtype='bool')

    def get_letter_idx(self, caption):
        # TODO(anguelos) make this work unicode. Maybe everything not in non letters low-ascii
        return np.array([c in string.letters for c in caption], dtype='bool')

    def split_textlines(self, caption, char_ltrb):
        """Split a textblock string into textline substrings given its boundinb boxes.

        :param caption:
        :param char_ltrb:
        :return:
        """
        visible_chars_idx = self.get_visible_idx(caption)
        line_start = 0
        texline_index_ranges = []
        while line_start < len(visible_chars_idx):
            line_end = line_start + 1
            # while line_end<len(visible_chars_idx) and char_ltrb[line_start,1]== char_ltrb[line_end,1]:
            while line_end < len(visible_chars_idx) and char_ltrb[
                line_end - 1, 2] < char_ltrb[line_end, 0]:
                # while line_end < len(visible_chars_idx) and char_ltrb[line_end - 1, 3] > char_ltrb[line_end, 1]:
                line_end += 1

            # Striping the textline range from whitespace.
            striped_start, striped_end = line_start, line_end
            while striped_start < len(visible_chars_idx) and not \
            visible_chars_idx[striped_start]:
                striped_start += 1
            while striped_end > striped_start and not visible_chars_idx[
                striped_end - 1]:
                striped_end -= 1
            texline_index_ranges.append([striped_start, striped_end])

            line_start = line_end
            while line_start < len(visible_chars_idx) and not visible_chars_idx[
                line_start]:
                line_start += 1
        res = np.array(texline_index_ranges, dtype='int32')
        res = res[res[:,0]<res[:,1]] # TODO(anguelos) this should not be necessary
        return res

    def split_words(self, caption, char_ltrb):
        """Split a textblock string into word substrings given its boundinb boxes. Punctuation is ignored.

        :param caption:
        :param char_ltrb:
        :return:
        """
        letter_chars_idx = self.get_letter_idx(caption)
        word_index_ranges = []
        word_start = 0
        # clearing possible non letters in the beginning of the line
        while word_start < len(letter_chars_idx) and not letter_chars_idx[
            word_start]:
            word_start += 1
        while word_start < len(letter_chars_idx):
            word_end = word_start + 1
            # while char is letter and and the bbox top is the same we are in the same word.
            while word_end < len(letter_chars_idx) and letter_chars_idx[
                word_end] and char_ltrb[word_start, 2] < char_ltrb[word_end, 0]:
                word_end += 1
            word_index_ranges.append([word_start, word_end])
            word_start = word_end
            while word_start < len(letter_chars_idx) and not letter_chars_idx[
                word_start]:
                word_start += 1
        return np.array(word_index_ranges, dtype='int32')

    def split_letters(self, caption, char_ltrb):
        """Split a textblock string into grapheme substrings given its boundinb boxes. Whitespace is ignored.

        :param caption:
        :param char_ltrb:
        :return:
        """
        letter_chars_idx = self.get_visible_idx(caption).nonzero()[0]
        res = np.empty([letter_chars_idx.shape[0], 2], dtype='int32')
        res[:, 0] = letter_chars_idx
        res[:, 1] = letter_chars_idx + 1
        return res

    def stich_ranges(self, caption, range_array, char_ltrb):
        """Takes a string, it's respective bounding boxes, and ranges of all the substrings and provides bounding boxes
        of the substrings.

        :param caption: A string of length N
        :param range_array: A list of tuples containing tuples with the substring beginings and ends of length M
        :param char_ltrb: An int32 numpy array of size [N,4]
        :return: A tuple with a numpy array of size [M] of objects containing the substrings and an int32 numpy array
        of size [M,4] containing the LTRB bounding boxes of the respective substrings.
        """
        range_captions = np.empty(range_array.shape[0], dtype='object')
        range_ltrb = np.empty([range_array.shape[0], 4], dtype='int32')

        if self.split_letters== self.split_substrings:
            print "Letters"
        elif self.split_textlines== self.split_substrings:
            print "Textlines"
        for n, ranges in enumerate(range_array):
            range_captions[n] = caption[ranges[0]:ranges[1]]
            range_ltrb[n, :] = char_ltrb[ranges[0]:ranges[1],
                               0].min(), char_ltrb[ranges[0]:ranges[1],
                                         1].min(), \
                               char_ltrb[ranges[0]:ranges[1],
                               2].max(), char_ltrb[ranges[0]:ranges[1], 3].max()
        return range_captions, range_ltrb

    @staticmethod
    def generate_masks(self, out_width, out_height, ltrb, annotations=None,
                       bin_mask=False):
        # This only works when no overlaps are guaratied of
        ltrb = ltrb.copy()
        ltrb[:, 2:] += 1
        if annotations is None:
            annotations = np.ones(bboxes.shape[0], dtype="int32")
        grad_img = np.zeros([out_height, out_width], dtype="int32")
        grad_img[ltrb[:, 1], ltrb[:, 0]] = grad_img[ltrb[:, 1],
                                                    ltrb[:, 0]] + annotations
        grad_img[ltrb[:, 3], ltrb[:, 2]] = grad_img[ltrb[:, 3],
                                                    ltrb[:, 2]] + annotations
        grad_img[ltrb[:, 1], ltrb[:, 2]] = grad_img[ltrb[:, 1],
                                                    ltrb[:, 2]] - annotations
        grad_img[ltrb[:, 3], ltrb[:, 0]] = grad_img[ltrb[:, 3],
                                                    ltrb[:,0]] - annotations
        result_mask = grad_img.cumsum(axis=0).cumsum(axis=1)
        if bin_mask:
            result_mask = (result_mask > 0).astype("int32")
        return result_mask

    def crop_page_boxes(self, gt_str_list=None, dilate=.5):
        dilate = int(dilate * self.letter_height)
        img, bboxes = self.current_img, self.current_roi_ltrb.astype("int32")
        if gt_str_list is None:
            gt_str_list = self.current_roi_captions
        box_l = bboxes[:, 0] - dilate
        box_l[box_l < 0] = 0
        box_t = bboxes[:, 1] - dilate
        box_t[box_t < 0] = 0
        box_r = bboxes[:, 2] + dilate
        box_r[box_r >= img.shape[1]] = img.shape[1]
        box_b = bboxes[:, 3] + dilate
        box_b[box_b >= img.shape[0]] = img.shape[0]
        img_list = []
        byte_img = (img * 255).astype("uint8")
        caption_list = []
        for n, ltrb in enumerate(bboxes):
            img_list.append(byte_img[box_t[n]:box_b[n], box_l[n]:box_r[n]])
            caption_list.append(gt_str_list[n])
        return img_list, caption_list

    def save_end2end_gt(self,image_fname,gt_fname,gt_accumulate_fname,rr2013_format=False):
        save_image_float(self.current_img,image_fname)
        string_table=[]
        ltrb=self.current_roi_ltrb.tolist()
        for k in range(ltrb.shape[0]):
            string_table.append([str(col) for col in ltrb[k]+[repr(self.current_roi_captions[k])]])
        if gt_accumulate_fname is not None:
            fname_string_table=[[repr(image_fname)]+str_line for str_line in string_table]
            gt_accumulate_tbl="\n".join(["\t".join(line) for line in fname_string_table])
            open(gt_accumulate_fname,"a")

        if gt_fname is not None:
            if rr2013_format:
                separator=","
            else:
                separator="\t"
            gt_tbl = "\n".join(
                [separator.join(line) for line in fname_string_table])
            open(gt_fname, "w").write(gt_tbl)

def demo_printed(corpus_txt_fname=None, quantum="textlines", plot_page=False,
             img_path_expr='/tmp/{}_{}.png', max_pages_count=1,
             letter_height=30):
    if corpus_txt_fname is None:
        corpus = lm_util.OcrCorpus.create_iliad_corpus(lang="eng")
    else:
        corpus = lm_util.OcrCorpus.create_file_corpus(corpus_txt_fname)
    synth = Synthesizer.create_printed_synthecizer(quantum=quantum,
                                               letter_height=letter_height,
                                               caption_reader=corpus)

    if plot_page:
        synth.plot_current_page()
    page_counter = 0
    gt_expr = unicode(img_path_expr.split("/")[-1] + '\t"{}"\n')
    # gt_path_expr = save_sample_expression + ".gt.txt"
    gt_dir = img_path_expr[:img_path_expr.rfind("/")]
    gt_accumulation = codecs.open((gt_dir + "/{}.gt.txt").format(quantum),
                                  mode="w", encoding="utf-8")
    codecs.open((gt_dir + "/{}.map.tsv").format(quantum), mode="w",
                encoding="utf-8").write(
        synth.caption_generator.get_alphabet_tsv())
    sample_counter = 0
    while True and img_path_expr:
        synth.generate_new_page()
        gt_img, gt_captions = synth.crop_page_boxes()
        print "Page ", page_counter, " Storing ", len(gt_img), " ", quantum
        for n in range(len(gt_img)):
            img_path = img_path_expr.format(quantum, sample_counter + n)
            gt_line = gt_expr.format(quantum, sample_counter + n,
                                     ' '.join(gt_captions[n].split()))
            # codecs.open(gt_path,mode= "w",encoding="utf-8").write(gt_line)
            gt_accumulation.write(gt_line)
            save_image_float(gt_img[n],img_path)
        sample_counter += len(gt_img)
        page_counter += 1
        if page_counter > int(max_pages_count):
            break
        else:
            print (page_counter, max_pages_count)


def demo_handwriting(corpus_txt_fname=None, quantum="textlines",
                     plot_page=False, img_path_expr='/tmp/sample_ds/{}_{}.png',
                     max_pages_count=1, letter_height=30,bg_img_pattern="./data/backgrounds/*.jpg",
                     font_names=("Pacifico","Cookie","Gaegu","Sacramento","Tangerine","Allura")):
    src_dir, _ = os.path.split(__file__)
    if corpus_txt_fname is None:
        try:
            corpus = lm_util.OcrCorpus.create_iliad_corpus(lang="eng")
        except:
            duckling_path = os.path.join(src_dir, "data", "corpora", "01_the_ugly_duckling.txt")
            corpus = lm_util.OcrCorpus.create_file_corpus(duckling_path)
    else:
        corpus = lm_util.OcrCorpus.create_file_corpus(corpus_txt_fname)
    out_dir, _ = os.path.split(img_path_expr)
    mkdir_p(out_dir)

    bg_paths = glob.glob(bg_img_pattern)
    print "BG:",bg_paths
    synth = Synthesizer.create_handwriten_synthecizer(quantum=quantum,letter_height=letter_height,caption_reader=corpus,bg_img_list=bg_paths,font_names=font_names)

    page_counter = 0
    gt_expr = unicode(img_path_expr.split("/")[-1] + '\t"{}"\n')
    # gt_path_expr = save_sample_expression + ".gt.txt"
    gt_dir = img_path_expr[:img_path_expr.rfind("/")]
    gt_accumulation = codecs.open((gt_dir + "/{}.gt.txt").format(quantum),
                                  mode="w", encoding="utf-8")
    codecs.open((gt_dir + "/{}.map.tsv").format(quantum), mode="w",
                encoding="utf-8").write(
        synth.caption_generator.get_alphabet_tsv())
    sample_counter = 0
    while True and img_path_expr:
        synth.generate_new_page()
        save_image_float(synth.gt_img,img_path_expr.format("gt",page_counter))
        save_image_float(synth.final_img, img_path_expr.format("clr", page_counter))

        if plot_page:
            synth.plot_current_page()
        gt_img, gt_captions = synth.crop_page_boxes()
        print "Page ", page_counter, " Storing ", len(gt_img), " ", quantum
        for n in range(len(gt_img)):
            img_path = img_path_expr.format(quantum, sample_counter + n)
            gt_line = gt_expr.format(quantum, sample_counter + n,
                                     ' '.join(gt_captions[n].split()))
            gt_accumulation.write(gt_line)
            save_image_float(gt_img[n], img_path)
        sample_counter += len(gt_img)
        page_counter += 1
        if page_counter > int(max_pages_count):
            break
        else:
            print (page_counter, max_pages_count)
