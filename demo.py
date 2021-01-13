import os
import cv2
import time
import numpy as np
from tqdm import tqdm

from algorithm import Matting, cutout, composite

if __name__ == '__main__':
    img_dir         = 'D:/Mission/photos'
    out_comp_dir    = 'out/comp'
    out_matte_dir   = 'out/matte'
    out_trimap_dir  = 'out/trimap'
    out_cutout_dir  = 'out/cutout'
    checkpoint_path = 'algorithm/matte/ckpt/best.pt'

    bg_color = [33, 150, 243]  # BGR

    os.makedirs(out_comp_dir, exist_ok=True)
    os.makedirs(out_trimap_dir, exist_ok=True)
    os.makedirs(out_matte_dir, exist_ok=True)
    os.makedirs(out_cutout_dir, exist_ok=True)

    M = Matting(checkpoint_path=checkpoint_path, gpu=True)

    img_names = os.listdir(img_dir)

    for name in tqdm(img_names):
        img_path = os.path.join(img_dir, name)

        matte, img, trimap = M.matting(img_path, with_img_trimap=True, net_img_size=480, max_size=378)

        start_time = time.time()
        cut = cutout(img, matte)
        comp = composite(cut, np.array(bg_color) / 255.)

        name = name.replace('.jpg', '.png')

        cv2.imwrite(os.path.join(out_matte_dir, name), np.uint8(matte * 255))
        cv2.imwrite(os.path.join(out_trimap_dir, name), np.uint8(trimap * 255))
        cv2.imwrite(os.path.join(out_comp_dir, name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_cutout_dir, name), cv2.cvtColor(np.uint8(cut * 255), cv2.COLOR_RGBA2BGRA))
