import os

import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex, QSemaphore

from algorithm import Matting, cutout, composite
from controller.singleton import Singleton
from model.task import Task, TaskState
from params import const
from enum import Enum
import numpy as np


class ModelState(Enum):
    FAIL         = -1
    INITIAL      = 0
    LOADED       = 1


@Singleton
class Controller:
    m: Matting = None
    model_state = ModelState.INITIAL
    tasks = []
    task_callbacks = []
    finished_task_num = 0
    transparent_bg = None
    # matting_lock = QMutex()
    matting_lock = QSemaphore(4)

    def __init__(self):
        h = w = const.MAX_SIZE
        block_width = 8
        chess = np.ones((((round(h / block_width)) + 1) * block_width, ((round(w / block_width)) + 1) * block_width, 1)) * 180
        white_block = np.full((block_width, block_width, 1), 235)
        for row in range(round(h / block_width)):
            for col in range(round(w / block_width)):
                if (row + col) % 2 == 0:
                    row_begin = row * block_width
                    row_end = row_begin + block_width
                    col_begin = col * block_width
                    col_end = col_begin + block_width
                    chess[row_begin:row_end, col_begin:col_end] = white_block
        self.transparent_bg = chess

    def load(self) -> bool:
        # time-consuming operation
        try:
            self.m = Matting(model_path=const.MODEL_PATH, model_fix_path=const.MODEL_FIX_PATH, gpu=True)
            self.model_state = ModelState.LOADED
            return True   # succeed
        except Exception:
            self.model_state = ModelState.FAIL
            return False  # failed

    def add_tasks(self, img_urls):
        new_tasks = []
        for url in img_urls:
            task = Task(url)
            self.tasks.append(task)
            new_tasks.append(task)
        return new_tasks

    def add_task_callback(self, callback):
        self.task_callbacks.append(callback)

    def add_task_callbacks(self, callbacks):
        new_callbacks = []
        for c in callbacks:
            new_callbacks.append(c)
        self.task_callbacks.append(new_callbacks)
        return new_callbacks

    def save_results(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for t in self.tasks:
            name = t.name
            if t.state != TaskState.COMPING_DONE:
                continue
            if t.bg is None:
                # cutout must use png.
                name = name.replace('.jpg', '.png')
                name = name.replace('.jpeg', '.png')
                name = name.replace('.jpe', '.png')
                name = name.replace('.JPG', '.png')
                name = name.replace('.JPEG', '.png')
                name = name.replace('.JPE', '.png')
                cv2.imwrite(os.path.join(save_dir, name), cv2.cvtColor(np.uint8(t.cutout), cv2.COLOR_RGBA2BGRA))
            else:
                cv2.imwrite(os.path.join(save_dir, name), cv2.cvtColor(np.uint8(t.comp), cv2.COLOR_RGB2BGR))

    @staticmethod
    def exec_tasks(tasks, callbacks):
        pass
        # for t, c in zip(tasks, callbacks):
        #     print(t.orig_path)

    def matting(self, img_url, bg=None, trimap=None):
        # time-consuming operation
        self.matting_lock.acquire()
        matte, img, trimap = self.m.matting(img_url, with_img_trimap=True, net_img_size=480, max_size=const.MAX_SIZE, trimap=trimap)
        if bg is None:
            h, w, c = img.shape
            bg = self.transparent_bg[:h, :w]

        cut = cutout(img, matte)
        comp = composite(cut, bg / 255.)
        self.matting_lock.release()
        self.finished_task_num += 1
        return np.uint8(matte * 255), np.uint8(cut * 255), np.uint8(comp * 255), np.uint8(trimap * 255)

    def change_background(self, task, bg=None):
        cut = task.cutout / 255.
        task.bg = bg
        if bg is None:
            h, w, c = cut.shape
            bg = self.transparent_bg[:h, :w]
        comp = composite(cut, bg / 255.)
        return comp * 255
