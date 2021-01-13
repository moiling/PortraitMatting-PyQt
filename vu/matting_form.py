import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget

from model.task import TaskState
from resource.matting_form import Ui_MattingForm
import numpy as np


class MattingForm(Ui_MattingForm, QWidget):
    def __init__(self, task):
        super().__init__()
        self.setupUi(self)
        self.task = task
        self.img_url = task.orig_path
        self.show_origin_image()

    def show_origin_image(self):
        pix = QPixmap(self.img_url)
        self.image_label.setPixmap(pix)

    @pyqtSlot()
    def result_callback(self):
        print('callback')
        # get task result img and update.
        if self.task.state != TaskState.COMPING_DONE:
            return
        self.show_result_image()

    def show_result_image(self):
        comp = self.task.comp.copy().astype(np.uint8)
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)

        print(comp)

        height, width, depth = comp.shape
        q_image = QImage(comp.data, width, height, width * depth, QImage.Format_RGB888)
        pix = QPixmap(q_image)

        print(pix)
        print(self.result_label)
        # TODO: why here can't update UI?
        self.result_label.setText('a')
