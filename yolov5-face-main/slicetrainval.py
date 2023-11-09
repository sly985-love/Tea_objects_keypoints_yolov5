from pathlib import Path
import random
from shutil import copy2

# target_path = Path(r'D:\Cp\pycharm\tea\tea_data')
# label_root = Path(r'D:\Cp\pycharm\tea\tea_data\labels')
# image_root = Path(r'D:\Cp\pycharm\tea\tea_data\images_640')

target_path = Path(r'D:\yolotea6\new_data')
label_root = Path(r'D:\yolotea6\new_data\all_label')
image_root = Path(r'D:\yolotea6\new_data\all_new')

# slice_data = [0.8, 0.2]
slice_data = [0.8, 0.1, 0.1]

label_list = [path for path in label_root.iterdir()]
llist_len = len(label_list)
llist_index = list(range(llist_len))
random.shuffle(llist_index)

train_folder = target_path / 'train'
val_folder = target_path / 'val'
test_folder = target_path / 'test'

train_stop_flag = llist_len * slice_data[0]
val_stop_flag = llist_len * (slice_data[0] + slice_data[1])
test_stop_flag = llist_len * (slice_data[0] + slice_data[1] + slice_data[2])

current_idx = 0
train_num = 0
val_num = 0
test_num = 0
for j, i in enumerate(llist_index):
    current_idx = i
    src_label_path = label_list[i]
    img_path = image_root / (label_list[i].stem + '.jpg')

    if j <= train_stop_flag:
        copy2(src_label_path, train_folder)
        copy2(img_path, train_folder)
        print("{}复制到了{}".format(img_path, train_folder))
        train_num = train_num + 1
    # elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
    elif (j <= val_stop_flag) and (j > train_stop_flag):
        copy2(src_label_path, val_folder)
        copy2(img_path, val_folder)
        print("{}复制到了{}".format(img_path, val_folder))
        val_num = val_num + 1
        # print("{}复制到了{}".format(src_img_path, test_folder))
    else:
        copy2(src_label_path, test_folder)
        copy2(img_path, test_folder)
        print("{}复制到了{}".format(img_path, test_folder))
        test_num = test_num + 1
        # print("{}复制到了{}".format(src_img_path, test_folder))
