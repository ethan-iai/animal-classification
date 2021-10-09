import os
import random
import shutil

def main():
    pwd = os.path.dirname(os.path.abspath(__file__))
    src_base = os.path.join(pwd, "train")
    pwd = os.path.join(pwd, "data")
    os.mkdir(pwd)
    dst2_base = os.path.join(pwd, "train")
    dst_base = os.path.join(pwd, "val")
    if not os.path.exists(pwd):
        os.mkdir(pwd)
    if not os.path.exists(dst2_base):
        os.mkdir(dst2_base)
    if not os.path.exists(dst_base):
        os.mkdir(dst_base)

    for sub_dir in sorted(os.listdir(src_base)):
        src_dir = os.path.join(src_base, sub_dir)
        dst_dir = os.path.join(dst_base, sub_dir)
        dst2_dir = os.path.join(dst2_base, sub_dir)
        os.makedirs(dst_dir)
        os.makedirs(dst2_dir)
        image_names = os.listdir(os.path.join(src_base, sub_dir))
        random.shuffle(image_names)        
        
        for name in image_names[: int(0.2 * len(image_names))]:
            shutil.copy2(os.path.join(src_dir, name), dst_dir)
        for name in image_names[int(0.2 * len(image_names)):]:
            shutil.copy2(os.path.join(src_dir, name), dst2_dir)    

if __name__ == '__main__':
    main()