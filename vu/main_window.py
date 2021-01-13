from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal

from controller.controller import Controller, ModelState
from resource.main_window import Ui_MainWindow
from vu.matting_form import MattingForm


class MainWindow(Ui_MainWindow, QMainWindow):

    matting_forms = []

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.m_flag, self.m_Position = False, None
        self.set_click_listener()

        self.setAcceptDrops(True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.init()

    def init(self):
        self.hint.setText('MODEL HAS NOT BEEN LOADED.')

    @pyqtSlot(bool)
    def load_model_callback(self, succeed):
        if succeed:
            self.hint.setHidden(True)
        else:
            self.hint.setText('MODEL LOAD FAILED!')

    def set_click_listener(self):
        self.close_button.clicked.connect(self.on_close_button_click)
        self.mini_button.clicked.connect(self.on_mini_button_click)
        self.hint.clicked.connect(self.on_hint_click)

    @pyqtSlot()
    def on_hint_click(self):
        if Controller().model_state is not ModelState.LOADED:
            self.hint.setText('LOADING MODEL...')
            Controller().load_model(self.load_model_callback)

    def dragEnterEvent(self, evn):
        evn.accept()

    def dropEvent(self, evn: QtGui.QDropEvent):
        if Controller().model_state == ModelState.LOADED:
            file_urls = evn.mimeData().text().strip().split('\n')
            # image file(.jpg, .jpeg, .png) only.
            img_urls = []
            for url in file_urls:
                if url[:7] != 'file://':
                    continue
                if url.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                    continue

                img_urls.append(url[8:] if url[9] == ':' else url[7:])  # macOS & windows

            # create tasks and matting forms.
            tasks = Controller().add_tasks(img_urls)
            callbacks = []
            for t in tasks:
                form = MattingForm(t)
                callback = form.result_callback
                Controller().add_task_callback(callback)
                callbacks.append(callback)
                self.content.layout().addWidget(MattingForm(t))

            Controller().exec_tasks(tasks, callbacks)
            print(img_urls)

    @pyqtSlot()
    def on_close_button_click(self):
        self.close()

    @pyqtSlot()
    def on_mini_button_click(self):
        self.showMinimized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.m_flag:
            self.move(event.globalPos() - self.m_Position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))
