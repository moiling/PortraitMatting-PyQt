import os
import threading

import cv2
import time
import numpy as np

from algorithm import Matting, cutout, composite


def matting(img_dir, out_comp_dir, bg_color, img_names):
    for name in img_names:
        print(f'start: {name}')
        img_path = os.path.join(img_dir, name)

        matte, img, trimap = M.matting(img_path, with_img_trimap=True, net_img_size=480, max_size=378)

        start_time = time.time()
        cut = cutout(img, matte)
        comp = composite(cut, np.array(bg_color) / 255.)

        name = name.replace('.jpg', '.png')

        print(f'end: {name}')

        cv2.imwrite(os.path.join(out_comp_dir, name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    img_dir         = 'D:/Mission/photos'
    out_comp_dir    = 'out/comp'
    checkpoint_path = 'algorithm/matte/ckpt/best.pt'

    bg_color = [33, 150, 243]  # BGR

    os.makedirs(out_comp_dir, exist_ok=True)

    M = Matting(checkpoint_path=checkpoint_path, gpu=True)

    img_names = os.listdir(img_dir)

    cut_size = 10
    img_name_cut = np.array_split(img_names, cut_size)

    ths = []
    for img_names in img_name_cut:
        ths.append(threading.Thread(target=matting, args=(img_dir, out_comp_dir, bg_color, img_names)))

    for th in ths:
        th.start()

