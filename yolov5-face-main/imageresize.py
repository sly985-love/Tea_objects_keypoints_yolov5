from pathlib import Path
from PIL import Image

# path_root = Path('D:\Program\python\\tea\yolov5-master\VOCdevkit')
path_root = Path(r'D:\Cp\pycharm\tea\tea_data')
path_imgs = path_root / 'JPEGImages'
path_new = path_root / 'images_640'
need = 640
if not path_imgs.exists():
    path_imgs.mkdir()

if not path_new.exists():
    path_new.mkdir()

list_path = [path for path in path_imgs.iterdir()]

# for path_img in list_path:
#     sta = path_img.stem
#
#     if not (path_new / sta).exists():
#         (path_new / sta).mkdir()

    # list_img = [path for path in path_img.iterdir()]

for imagep in list_path:
    name = imagep.stem
    img = Image.open(imagep)
    w, h = img.size
    wn, hn = w//need, h//need
    iw = min(wn, hn)
    img_new = img.resize((w//iw, h//iw))
    new_path = path_new / (name + '.jpg')
    img_new.save(new_path)