import math
from threading import current_thread

from PIL import Image, ImageQt
from torchvision.transforms import functional as F
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget

from controller.controller import Controller
from model.task import TaskState
from params import const
from resource.matting_form import Ui_MattingForm
import numpy as np


class MattingThread(QThread):
    matting_finished_trigger = pyqtSignal()
    thread = None

    def __init__(self, task, bg=None):
        super().__init__()
        self.task = task
        self.img_url = task.orig_path
        self.bg = bg

    def run(self):
        self.task.alpha, self.task.cutout, self.task.comp = Controller().matting(self.img_url, self.bg)
        self.task.state = TaskState.COMPING_DONE
        self.matting_finished_trigger.emit()


class MattingForm(Ui_MattingForm, QWidget):
    def __init__(self, task):
        super().__init__()
        self.setupUi(self)
        self.task = task
        self.img_url = task.orig_path
        self.show_origin_image()

    def matting(self):
        self.thread = MattingThread(self.task)
        self.thread.matting_finished_trigger.connect(self.result_callback)
        self.thread.start()
        # self.thread.exec()  # can't use exec!

    def show_origin_image(self):
        size = const.MAX_SIZE
        img = Image.open(self.img_url)
        max_size = max(img.size[0], img.size[1])
        if max_size > size:
            rate = size / float(max_size)
            h, w = math.ceil(rate * img.size[0]), math.ceil(rate * img.size[1])
            img = F.resize(img, [w, h])

        pix = ImageQt.toqpixmap(img)
        self.image_label.setPixmap(pix)
        self.img_name.setText(self.task.name)

    @pyqtSlot()
    def result_callback(self):
        # get task result img and update.
        self.show_result_image()

    def show_result_image(self):
        if self.task.state != TaskState.COMPING_DONE:
            return

        comp = self.task.comp.copy().astype(np.uint8)

        height, width, depth = comp.shape
        q_image = QImage(comp.data, width, height, width * depth, QImage.Format_RGB888)
        pix = QPixmap(q_image)

        self.result_label.setPixmap(pix)
