import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from algorithm import Matting, cutout, composite
from controller.singleton import Singleton
from model.task import Task, TaskState
from params import const
from enum import Enum
import numpy as np


class LoadModelThread(QThread):
    load_finished_trigger = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

    def run(self):
        succeed = Controller().load()
        self.load_finished_trigger.emit(succeed)


class MattingThread(QThread):
    matting_finished_trigger = pyqtSignal()

    def __init__(self, task, bg=None):
        super().__init__()
        self.task = task
        self.img_url = task.orig_path
        self.bg = bg

    def run(self):
        print('run')
        # self.task.alpha, self.task.cutout, self.task.comp = Controller().matting(self.img_url, self.bg)
        self.task.comp = cv2.imread(self.img_url)
        self.task.state = TaskState.COMPING_DONE
        print('done')
        self.matting_finished_trigger.emit()


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

    def __init__(self):
        pass

    @staticmethod
    def load_model(callback):
        thread = LoadModelThread()
        thread.load_finished_trigger.connect(callback)
        thread.start()
        thread.exec()

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
        # for t, c in zip(tasks, callbacks):
        #     print(t.orig_path)
        thread = MattingThread(tasks[0])
        thread.matting_finished_trigger.connect(callbacks[0])
        # thread.start()
        # thread.exec()
        thread.run()

    def matting(self, img_url, bg=None):
        # time-consuming operation
        if not bg:
            bg = np.array([128, 128, 128])

        matte, img, trimap = self.m.matting(img_url, with_img_trimap=True, net_img_size=480, max_size=378)
        cut = cutout(img, matte)
        comp = composite(cut, bg / 255.)
        return matte, cut, comp
