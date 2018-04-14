from commands import getoutput as shell_stdout
import os

def mkdir_p(path):
    shell_stdout("mkdir -p "+path)

def resumable_download(url,save_dir):
    mkdir_p(save_dir)
    download_cmd = 'wget --directory-prefix=%s -c %s' % (save_dir, url)
    print "Downloading %s ... "%url,
    shell_stdout(download_cmd)
    print "done"
    return os.path.join(save_dir,url.split("/")[-1])