from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import math
from enum import Enum
import numpy as np


class PenSize(Enum):
    SMALL = 10
    MIDDLE = 20
    LARGE = 30


class PenType(Enum):
    FOREGROUND = 0
    BACKGROUND = 1
    ERASER = 2


class DrawTrimapLabel(QLabel):
    def __init__(self, *__args, width, height, init_trimap):
        super().__init__(*__args)
        self.endPoint = QPoint()
        self.lastPoint = QPoint()
        self.painter = QPainter()
        self.pix_width = width
        self.pix_height = height
        self.pix_left, self.pix_top = 0, 0
        h, w, c = init_trimap.shape
        init_trimap = init_trimap.squeeze()
        trimap_show = np.zeros((h, w, 3), dtype=np.uint8)
        trimap_show[:, :, 0] = (init_trimap == 255) * 255
        trimap_show[:, :, 2] = (init_trimap == 0) * 255
        q_image = QImage(trimap_show.data, w, h, w * 3, QImage.Format_RGB888)
        self.pix = QPixmap(q_image)
        # self.pix = QPixmap(width, height)
        # self.pix.fill(Qt.transparent)
        self.pen_type = PenType.FOREGROUND
        self.pen_size = PenSize.LARGE
        # self.setStyleSheet('QLabel{opacity:0.5;}')
        op = QtWidgets.QGraphicsOpacityEffect()
        op.setOpacity(0.5)
        self.setGraphicsEffect(op)

    def reset_pix(self, width, height, init_trimap):
        self.endPoint = QPoint()
        self.lastPoint = QPoint()
        self.painter = QPainter()
        self.pix_width = width
        self.pix_height = height
        self.pix_left, self.pix_top = 0, 0
        h, w, c = init_trimap.shape
        init_trimap = init_trimap.squeeze()
        trimap_show = np.zeros((h, w, 3), dtype=np.uint8)
        trimap_show[:, :, 0] = (init_trimap == 255) * 255
        trimap_show[:, :, 2] = (init_trimap == 0) * 255
        q_image = QImage(trimap_show.data, w, h, w * 3, QImage.Format_RGB888)
        self.pix = QPixmap(q_image)

    def paintEvent(self, event):
        self.pix_left = (self.width() - self.pix_width) // 2
        self.pix_top = (self.height() - self.pix_height) // 2
        self.painter.begin(self)
        self.painter.drawPixmap(self.pix_left, self.pix_top, self.pix)
        self.painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.endPoint = event.pos()
            self.painter.begin(self.pix)
            if self.pen_type == PenType.BACKGROUND:
                self.painter.setPen(QPen(QColor(0, 0, 255), self.pen_size.value))
            if self.pen_type == PenType.FOREGROUND:
                self.painter.setPen(QPen(QColor(255, 0, 0), self.pen_size.value))
            if self.pen_type == PenType.ERASER:
                self.painter.setPen(QPen(QColor(0, 0, 0), self.pen_size.value))
            self.painter.drawLine(self.lastPoint.x() - self.pix_left, self.lastPoint.y() - self.pix_top, self.endPoint.x() - self.pix_left, self.lastPoint.y() - self.pix_top)
            self.painter.end()
            self.update()
            self.lastPoint = self.endPoint

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()
