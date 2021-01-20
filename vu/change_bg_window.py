from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow

from controller.controller import Controller
from model.task import TaskState
from resource.change_bg_window import Ui_ChangeBgWindow
from vu.base_window import BaseWindow
import numpy as np


class ChangeBgThread(QThread):
    change_finished_trigger = pyqtSignal()
    thread = None

    def __init__(self, task, bg=None):
        super().__init__()
        self.task = task
        self.bg = bg

    def run(self):
        if self.task.state != TaskState.COMPING_DONE:
            return

        self.task.comp = Controller().change_background(self.task, self.bg)
        self.change_finished_trigger.emit()


class ChangeBgWindow(Ui_ChangeBgWindow, BaseWindow):
    def __init__(self, task, parent_change_bg_callback):
        super().__init__()
        self.setupUi(self)
        self.parent_change_bg_callback = parent_change_bg_callback
        self.task = task
        self.origin_bg = task.bg

        self.img_name.setText(self.task.short_name)
        self.set_listener()
        self.show_result_image()

    def set_listener(self):
        self.close_button.clicked.connect(self.on_cancel_button_click)
        self.ok_button.clicked.connect(self.on_close_button_click)
        self.mini_button.clicked.connect(self.on_mini_button_click)
        self.red_button.clicked.connect(self.on_red_button_click)
        self.blue_button.clicked.connect(self.on_blue_button_click)
        self.white_button.clicked.connect(self.on_white_button_click)
        self.clear_button.clicked.connect(self.on_clear_button_click)

    @pyqtSlot()
    def on_clear_button_click(self):
        self.change_bg()

    @pyqtSlot()
    def on_cancel_button_click(self):
        self.change_bg(self.origin_bg)
        self.on_close_button_click()

    @pyqtSlot()
    def on_red_button_click(self):
        self.change_bg(np.array([247, 102, 119]))

    @pyqtSlot()
    def on_blue_button_click(self):
        self.change_bg(np.array([85, 170, 255]))

    @pyqtSlot()
    def on_white_button_click(self):
        self.change_bg(np.array([255, 255, 255]))

    def change_bg(self, bg=None):
        self.thread = ChangeBgThread(self.task, bg)
        self.thread.change_finished_trigger.connect(self.show_result_image)
        self.thread.change_finished_trigger.connect(self.parent_change_bg_callback)
        self.thread.start()

    def show_result_image(self):
        if self.task.state != TaskState.COMPING_DONE:
            return

        comp = self.task.comp.copy().astype(np.uint8)

        height, width, depth = comp.shape
        q_image = QImage(comp.data, width, height, width * depth, QImage.Format_RGB888)
        pix = QPixmap(q_image)

        self.result_label.setPixmap(pix)


