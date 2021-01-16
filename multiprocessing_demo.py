import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from algorithm import Matting, cutout, composite

img_dir = 'D:/Mission/photos'
checkpoint_path = 'algorithm/matte/ckpt/best.pt'
out_comp_dir = 'out/comp'
bg_color = [33, 150, 243]  # BGR
M = Matting(checkpoint_path=checkpoint_path, gpu=False)


def matting(name):
    print(f'start: {name}')
    img_path = os.path.join(img_dir, name)

    matte, img, trimap = M.matting(img_path, with_img_trimap=True, net_img_size=480, max_size=378)
    cut = cutout(img, matte)
    comp = composite(cut, np.array(bg_color) / 255.)

    name = name.replace('.jpg', '.png')

    print(f'end: {name}')

    os.makedirs(out_comp_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_comp_dir, name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    start_time = time.time()

    thread_num = cv2.getNumberOfCPUs()
    pool = Pool(processes=10)
    pool.map(func=matting, iterable=img_names)
    pool.close()
    pool.join()

    print('total time:', time.time() - start_time)
