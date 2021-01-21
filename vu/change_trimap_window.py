import math

import cv2
from PIL import Image, ImageQt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, Qt, QRect, QThread, pyqtSignal

from controller.controller import Controller
from model.task import TaskState
from params import const
from resource import style
from resource.change_trimap_window import Ui_ChangeTrimapWindow
from vu.base_window import BaseWindow
import numpy as np
from torchvision.transforms import functional as F

from vu.widget.draw_trimap_label import DrawTrimapLabel, PenType, QImage, QPixmap, PenSize


class MattingThread(QThread):
    matting_finished_trigger = pyqtSignal()
    thread = None

    def __init__(self, task, trimap, bg=None):
        super().__init__()
        self.task = task
        self.img_url = task.orig_path
        self.trimap = trimap
        self.bg = bg

    def run(self):
        self.task.alpha, self.task.cutout, self.task.comp, self.task.trimap = Controller().matting(self.img_url, self.bg, self.trimap)
        self.task.state = TaskState.COMPING_DONE
        self.matting_finished_trigger.emit()


class ChangeTrimapWindow(Ui_ChangeTrimapWindow, BaseWindow):
    trimap_label: DrawTrimapLabel = None
    trimap = None

    def __init__(self, task, parent_callback):
        super().__init__()
        self.setupUi(self)
        self.task = task
        self.parent_callback = parent_callback
        self.orig_trimap = task.trimap
        self.set_listener()
        self.show_image()
        self.show_trimap()
        self.img_name.setText(self.task.short_name)

        self.pen_size_switch(PenSize.LARGE)
        self.tool_switch(PenType.FOREGROUND)

    def show_image(self):
        if self.task.state != TaskState.COMPING_DONE:
            return

        comp = self.task.comp.copy().astype(np.uint8)

        height, width, depth = comp.shape
        q_image = QImage(comp.data, width, height, width * depth, QImage.Format_RGB888)
        pix = QPixmap(q_image)
        self.image_label.setPixmap(pix)

    def show_trimap(self):
        height, width, _ = self.task.comp.shape

        size = const.MAX_SIZE
        self.trimap_label = DrawTrimapLabel(self.trimap_widget, width=width, height=height, init_trimap=self.task.trimap)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trimap_label.sizePolicy().hasHeightForWidth())
        self.trimap_label.setSizePolicy(sizePolicy)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trimap_label.setGeometry(QRect(10, 11, size, size))
        self.trimap_label.setObjectName('trimap_label')

    def refresh_trimap(self, trimap=None):
        if self.trimap_label is None:
            return

        height, width, _ = self.task.comp.shape
        if trimap is None:
            self.trimap_label.reset_pix(width=width, height=height, init_trimap=self.task.trimap)
        else:
            self.trimap_label.reset_pix(width=width, height=height, init_trimap=trimap)

    def set_listener(self):
        self.close_button.clicked.connect(self.on_cancel_button_click)
        self.mini_button.clicked.connect(self.on_mini_button_click)
        self.f_button.clicked.connect(self.on_fore_button_click)
        self.b_button.clicked.connect(self.on_back_button_click)
        self.e_button.clicked.connect(self.on_eraser_button_click)
        self.do_button.clicked.connect(self.on_do_button_click)
        self.small_button.clicked.connect(self.on_small_button_click)
        self.middle_button.clicked.connect(self.on_middle_button_click)
        self.large_button.clicked.connect(self.on_large_button_click)
        self.ok_button.clicked.connect(self.on_ok_button_click)
        self.clear_button.clicked.connect(self.on_clear_button_click)

    @pyqtSlot()
    def on_cancel_button_click(self):
        self.on_clear_button_click()
        self.on_close_button_click()

    @pyqtSlot()
    def on_ok_button_click(self):
        self.task.trimap = self.trimap
        self.on_close_button_click()

    @pyqtSlot()
    def on_clear_button_click(self):
        self.trimap = self.orig_trimap
        self.matting(self.trimap[..., 0])
        self.refresh_trimap(self.trimap)

    @pyqtSlot()
    def on_small_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_size = PenSize.SMALL
        self.pen_size_switch(PenSize.SMALL)

    @pyqtSlot()
    def on_middle_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_size = PenSize.MIDDLE
        self.pen_size_switch(PenSize.MIDDLE)

    @pyqtSlot()
    def on_large_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_size = PenSize.LARGE
        self.pen_size_switch(PenSize.LARGE)

    def pen_size_switch(self, size: PenSize):
        small = size == PenSize.SMALL
        middle = size == PenSize.MIDDLE
        large = size == PenSize.LARGE
        self.small_button.setStyleSheet(style.pen_small_current if small else style.pen_small_normal)
        self.middle_button.setStyleSheet(style.pen_middle_current if middle else style.pen_middle_normal)
        self.large_button.setStyleSheet(style.pen_large_current if large else style.pen_large_normal)

    def tool_switch(self, pen: PenType):
        fore = pen == PenType.FOREGROUND
        back = pen == PenType.BACKGROUND
        eraser = pen == PenType.ERASER
        # self.f_button.setStyleSheet(style.tool_button_current if fore else style.tool_button_normal)
        # self.b_button.setStyleSheet(style.tool_button_current if back else style.tool_button_normal)
        # self.e_button.setStyleSheet(style.tool_button_current if eraser else style.tool_button_normal)

    @pyqtSlot()
    def on_do_button_click(self):
        if self.trimap_label is None:
            return
        image = self.trimap_label.pix.toImage()
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
        arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
        trimap = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        trimap = cv2.resize(trimap, (self.task.trimap.shape[1], self.task.trimap.shape[0]), cv2.INTER_NEAREST)
        b = np.logical_and(np.logical_and(trimap[..., 0] > 128, trimap[..., 1] < 128), trimap[..., 2] < 128)
        f = np.logical_and(np.logical_and(trimap[..., 0] < 128, trimap[..., 1] < 128), trimap[..., 2] > 128)
        u = np.logical_not(np.logical_or(b, f))
        trimap[f] = [255, 255, 255]
        trimap[b] = [0, 0, 0]
        trimap[u] = [128, 128, 128]
        trimap = trimap[..., 0]
        self.trimap = trimap[..., np.newaxis]
        self.matting(trimap)

    def matting(self, trimap):
        self.thread = MattingThread(self.task, trimap, self.task.bg)
        self.thread.matting_finished_trigger.connect(self.result_callback)
        self.thread.matting_finished_trigger.connect(self.parent_callback)
        self.thread.start()

    @pyqtSlot()
    def result_callback(self):
        self.show_image()

    @pyqtSlot()
    def on_fore_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_type = PenType.FOREGROUND
        self.tool_switch(PenType.FOREGROUND)

    @pyqtSlot()
    def on_back_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_type = PenType.BACKGROUND
        self.tool_switch(PenType.BACKGROUND)

    @pyqtSlot()
    def on_eraser_button_click(self):
        if self.trimap_label is None:
            return
        self.trimap_label.pen_type = PenType.ERASER
        self.tool_switch(PenType.ERASER)
