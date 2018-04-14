# -*- coding: utf-8 -*-
import cairo
import pangocairo
import numpy as np
import pango

from dagtasets import mkdir_p,load_image_float,save_image_float


def box_filter1d(img, box_sz, horizontal=True, iter=1):
    # TODO (anguelos) add proper border options
    assert box_sz % 2 == 0 and box_sz > 0
    if horizontal:
        tmp_img = np.empty([img.shape[0], img.shape[1] + box_sz])
        tmp_img[:, box_sz / 2:-box_sz / 2] = img
        tmp_img[:, :box_sz / 2] = 0
        tmp_img[:, -box_sz / 2:] = 0

        div_map = np.ones(tmp_img.shape) * box_sz
        div_map[:, :box_sz] = np.ones([img.shape[0], box_sz]).cumsum(axis=1)
        div_map[:, -box_sz:] = np.ones(
            [img.shape[0], box_sz]).cumsum(axis=1)[:, ::-1]

        for _ in range(iter):
            new_img = np.empty(tmp_img.shape)
            new_img[:, :box_sz / 2] = 0
            new_img[:, -box_sz / 2:] = 0
            cs = tmp_img.cumsum(axis=1)
            new_img[:, box_sz / 2:-box_sz / 2] = (cs[:, box_sz:] -
                                                  cs[:, :-box_sz])
            new_img /= div_map
            tmp_img = new_img
        return new_img[:, box_sz / 2:-box_sz / 2]
    else:
        tmp_img = np.empty([img.shape[0] + box_sz, img.shape[1]]);
        tmp_img[box_sz / 2:-box_sz / 2, :] = img;
        tmp_img[:box_sz / 2, :] = 0;
        tmp_img[-box_sz / 2:, :] = 0;

        div_map = np.ones(tmp_img.shape) * box_sz
        div_map[:box_sz, :] = np.ones([box_sz, img.shape[1]]).cumsum(axis=0)
        div_map[-box_sz:, :] = np.ones([box_sz, img.shape[1]]).cumsum(axis=0)[
                               ::-1, :]

        for _ in range(iter):
            new_img = np.empty(tmp_img.shape)
            new_img[:box_sz / 2, :] = 0;
            new_img[-box_sz / 2:, :] = 0;
            cs = tmp_img.cumsum(axis=0)
            new_img[box_sz / 2:-box_sz / 2, :] = (
                        cs[box_sz:, :] - cs[:-box_sz, :])
            new_img /= div_map
            tmp_img = new_img
        return new_img[box_sz / 2:-box_sz / 2, :]


def fake_gaussian(img, vertical_horizontal_sigma, iter=3):
    """Gaussian filter aproximation with integrall images.

    :param img:
    :param vertical_horizontal_sigma:
    :param iter: An integer with the number of consecutive box filters used to approximate the gaussian kernel.
    :return: an image of the same size as img.
    """
    sigma_vertical, sigma_horizontal = vertical_horizontal_sigma
    h_blured = box_filter1d(img, sigma_horizontal, horizontal=True, iter=iter)
    blured = box_filter1d(h_blured, sigma_vertical, horizontal=False, iter=iter)
    return blured


def raw_interp2(X, Y, img):
    # TODO (anguelos) fix artifact Y instability seen on the spline interpolation
    e = .001
    X = X.copy();
    X[X < 0] = 0;
    X[X >= img.shape[1] - 2] = img.shape[1] - 2;
    Y = Y.shape[0]-Y;
    Y[Y < 0] = 0;
    Y[Y >= img.shape[0] - 2] = img.shape[0] - 2;

    left = np.floor(X)  # .astype('int32')
    right_coef = X - left
    left_coef = 1 - right_coef
    right = left + 1  # .astype('int32')

    top = np.floor(Y)  # .astype('int32')
    bottom_coef = Y - top
    top_coef = 1 - bottom_coef
    bottom = top + 1  # .astype('int32')

    left = left.astype("int32")
    top = top.astype("int32")
    right = right.astype("int32")
    bottom = bottom.astype("int32")

    res = np.empty(img.shape)
    if len(img.shape)==2:
        res[:] = img[top, left] * left_coef * top_coef + img[
            top, right] * right_coef * top_coef + \
                img[bottom, left] * left_coef * bottom_coef + img[
                    bottom, right] * right_coef * bottom_coef
    elif len(img.shape)==3:
        #res=res.reshape(-1,3)
        for c in range(3):
            cimg=img[:,:,c]
            res[:,:,c] = cimg[top, left] * left_coef * top_coef + cimg[top, right] * right_coef * top_coef + cimg[bottom, left] * left_coef * bottom_coef + cimg[bottom, right] * right_coef * bottom_coef
        res=res.reshape(img.shape)

    res[res < 0] = 0
    res[res > 1] = 1
    return res

def render_text(caption=np.array(list("Hello\nWorld")),fontname='times',font_height=30,image_width=1000,image_height=1000,crop_edge_ltrb=np.array([10,10,10,10]),alignment=pango.ALIGN_LEFT,crop_for_height=True):
    if isinstance(caption,str) or isinstance(caption,unicode):
        caption=np.array(list(caption))
    text_width = image_width - (crop_edge_ltrb[0] + crop_edge_ltrb[2])
    text_height = image_height - (crop_edge_ltrb[1] + crop_edge_ltrb[3])
    canvas_width = image_width #+ crop_edge_ltrb[0] + crop_edge_ltrb[2]
    canvas_height = image_height #+ crop_edge_ltrb[1] + crop_edge_ltrb[3]
    while True:
        surf = cairo.ImageSurface(cairo.FORMAT_A8, canvas_width, canvas_height)
        context = cairo.Context(surf)

        context.translate(crop_edge_ltrb[0], crop_edge_ltrb[1])
        pangocairo_context = pangocairo.CairoContext(context)
        layout = pangocairo_context.create_layout()
        layout.set_width(text_width * pango.SCALE)
        font = pango.FontDescription(fontname)
        font.set_absolute_size(font_height * pango.SCALE)
        layout.set_font_description(font)
        #print "FONT DESCRIPTION:", repr(font.get_family())
        layout.set_alignment(alignment)
        layout.set_wrap(pango.WRAP_WORD)
        layout.set_text(u" ".join(caption.tolist()))
        context.set_source_rgb(1, 0, 1.0)
        pangocairo_context.update_layout(layout)
        pangocairo_context.show_layout(layout)
        # pangocairo_context.paint()
        buf = surf.get_data()
        np_img = np.frombuffer(buf, np.uint8).reshape(
            [canvas_height, canvas_width])
        char_ltwh = np.array(
            [layout.index_to_pos(k) for k in range(len(caption))]) / pango.SCALE
        print char_ltwh.shape
        if crop_for_height and (char_ltwh[:,[1,3]].sum(axis=1)>text_height).all():
            inside_height= text_height >= char_ltwh[:, [1, 3]].sum(axis=1)
            if not inside_height.any():
                raise ValueError
            caption=caption[inside_height]
        else:
            break

    char_ltrb = char_ltwh.copy() + 1
    char_ltrb[:, 2:] = char_ltwh[:, 2:] + (
                char_ltwh[:, :2] - 2)  # guaranties no overlaps
    char_ltrb[:, [0, 2]] += (crop_edge_ltrb[0])
    char_ltrb[:, [1, 3]] += (crop_edge_ltrb[1])
    return 1-(np_img/255.0), char_ltrb
