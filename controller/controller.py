import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

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
    transparent_bg = None

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
            self.m = Matting(checkpoint_path=const.CHECKPOINT_PATH, gpu=True)
            self.model_state = ModelState.LOADED
            return True   # succeed
        except Exception:
            self.model_state = ModelState.FAIL
            return False  # failed

    def add_tasks(self, img_urls):
        new_tasks = []
        for url in img_urls:
            new_tasks.append(Task(url))
        self.tasks.append(new_tasks)
        return new_tasks

    def add_task_callback(self, callback):
        self.task_callbacks.append(callback)

    def add_task_callbacks(self, callbacks):
        new_callbacks = []
        for c in callbacks:
            new_callbacks.append(c)
        self.task_callbacks.append(new_callbacks)
        return new_callbacks

    @staticmethod
    def exec_tasks(tasks, callbacks):
        pass
        # for t, c in zip(tasks, callbacks):
        #     print(t.orig_path)

    def matting(self, img_url, bg=None):
        # time-consuming operation
        matte, img, trimap = self.m.matting(img_url, with_img_trimap=True, net_img_size=480, max_size=const.MAX_SIZE)
        if not bg:
            h, w, c = img.shape
            bg = self.transparent_bg[:h, :w]

        cut = cutout(img, matte)
        comp = composite(cut, bg / 255.)
        return matte * 255, cut * 255, comp * 255
