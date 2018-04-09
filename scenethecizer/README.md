# Demo

For the synthecizer to work it is assumed that 

List installed fonts:
```bash
synthecize -mode=listfonts
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
synthesize -help
```


Make a handwritten page and view it:
```bash
synthecize -mode=handwriten  -plot_page=True -quantum=words
``` 

Make a small textline handwritten dataset in /tmp/example_ds:
```bash
synthecize -mode=handwriten  -img_path_expr='/tmp/example_ds/{}_{}.png' -page_count=200
```