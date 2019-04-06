import numbers

def rgbstr(s, fg_rgb=(255, 255, 255), bg_rgb=(0,0,0) ):
    """Returns a colored coded string acording to unix terminal codes.

    :param s: The input string or unicode string.
    :param fg_rgb: The rgb value of the foreground as a tuple of integers from 0-255.
    :param bg_rgb: The rgb value of the background as a tuple of integers from 0-255.
    :return: A unicode screen with xterm escape codes to render the input string in the desired colors.
    """
    fr, fg, fb = fg_rgb
    br , bg , bb = bg_rgb
    return u"\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm%s\x1b[0m" % (fr, fg, fb, br, bg, bb, s)


def render_sequence(caption, labels=None, confidence=0, label_colors=((255, 0, 0),(0, 255, 0)),confidence_color=(255 ,255 ,255)):
    """Multiplexes categorical and confidence values into a string.

    This code is used to provide informative logs and visualising information on the CLI.
    Examples:

    An example of the use of this function for evaluating character generation is:
    pr = "hello world" # prediction
    pr_confidence = [.3, .7, .8 , .8, .7, .5, .1, .7, .6, .4, .8]
    gt = "Hello World" # groundtruth
    print(render_sequence(pr, [pr[k]==gt[k] for k in range(len(pr))], pr_confidence))

    A genomics example is:
    seq = "ACGTCCCGTTA"
    depth = [.9, .7, .7, .4, .3, .9, .3, .6, .1, 0.1, .7]
    SNIPs = [0, 4, 0, 1, 0, 0, 0, 2, 0, 0, 0]
    SNIPcolors = [(255, 255, 255), # No mutation
                  (0, 255, 255),   # A -> *
                  (0, 255, 0),     # C -> *
                  (255, 0, 0),     # G -> *
                  (128, 0, 255)]   # T -> *
    print(render_sequence(seq, SNIPs, depth, SNIPcolors))

    :param caption: The string to be rendered.
    :param labels: The catigorical variable that will dictate the foreground color. It must be either an integer or a collection of the same length as caption containing integers.
    :param confidence: The continuous variable that will dictate the background color. It must be either a numbe in the range [0, 1] or a collection  of the same length as caption containing such numbers.
    :param label_colors: The colors of each value labels can take. The colors are tuples byte-values in a collection.
    :param confidence_color: A tuple with byte-values that defines the rgb color for the background. The confidence is multiplied with those values for each string.
    :return: A unicode string with all the apropriate escape sequences to render the caption.
    """
    if labels is None:
        labels = (0,)*len(caption)
    elif isinstance(labels, int):
        labels = (labels,) * len(caption)

    if isinstance(confidence, numbers.Number):
        confidence = (confidence,) * len(caption)

    assert len(labels)==len(caption) and len(confidence)==len(caption)

    bg = [[int(255 * confidence[k]), int(255 * confidence[k]), int(255 * confidence[k])] for k in
          range(len(caption))]

    return u"".join([rgbstr(caption[k],label_colors[labels[k]],bg[k]) for k in range(len(caption))])


def render_prediction(caption, gt, confidence=0):
    correct=[caption[k]==gt[k] for k in range(len(caption))]
    return render_sequence(caption,correct,confidence)