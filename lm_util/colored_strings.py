def rgbstr(s, fg_rgb=(255, 255, 255), bg_rgb=(0,0,0) ):
    fr, fg, fb = fg_rgb
    br , bg , bb = bg_rgb
    return u"\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm%s\x1b[0m" % (fr, fg, fb, br, bg, bb, s)


def print_unicode_ssequense(caption, correct=None, confidence=0):
    try:
        if correct is None:
            fg = [[255, 255, 255]] * len(caption)
        else:
            fg = [[0, 255, 0] if correct[k] else [255, 0, 0] for k in range(len(caption))]
        try:
            bg = [[int(255 * confidence[k]), int(255 * confidence[k]), int(255 * confidence[k])] for k in
                  range(len(caption))]
        except:
            bg = [[int(255 * confidence), int(255 * confidence), int(255 * confidence)] for k in range(len(caption))]
        return u"".join([rgbstr(*([caption[k]] + fg[k] + bg[k])) for k in range(len(caption))])
    except:
        return caption