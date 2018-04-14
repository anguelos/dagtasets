from commands import getoutput as shell_stdout
import os
import errno
import cv2


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
    cv2.imwrite(fname,255-img)

