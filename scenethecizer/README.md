# Demo

For the synthecizer to work it is assumed that 

List installed fonts:
```bash
scenethesize -mode=listfonts
```

Install google fonts:
```bash
mkdir -p /tmp/gfonts
cd /tmp/gfonts
wget -c https://github.com/google/fonts/archive/master.zip
unzip -o *.zip
mkdir -p $HOME/.fonts
cp fonts-master/*/*/*ttf $HOME/.fonts
cd
rm -Rf  /tmp/gfonts
```

Print help:
```bash
scenethesize -help
```


Make a handwritten page and view it:
```bash
scenethesize -mode=handwriten  -plot_page=True -quantum=words
``` 

Make a small textline handwritten dataset in /tmp/example_ds:
```bash
scenethesize -mode=handwriten  -img_path_expr='/tmp/example_ds/{}_{}.png' -page_count=200
```

```bash
mkdir -p /tmp/unrealcv/
cd /tmp/unrealcv/
wget -c http://cs.jhu.edu/~qiuwch/release/unrealcv/RealisticRendering-Linux-0.3.10.zip
unzip -o RealisticRendering-Linux-0.3.10.zip
./RealisticRendering/LinuxNoEditor/playground.sh
#nvidia-docker run --name rr --rm -p 9000:9000 --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" qiuwch/rr:0.3.8
```