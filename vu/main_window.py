from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMainWindow, QLabel, QFileDialog
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal

from controller.controller import Controller, ModelState
from resource import style
from resource.main_window import Ui_MainWindow
from vu.base_window import BaseWindow
from vu.matting_form import MattingForm


class LoadModelThread(QThread):
    load_finished_trigger = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

    def run(self):
        succeed = Controller().load()
        self.load_finished_trigger.emit(succeed)


class SaveResultsThread(QThread):
    save_finished_trigger = pyqtSignal()

    def __init__(self, save_dir):
        self.save_dir = save_dir
        super().__init__()

    def run(self):
        Controller().save_results(self.save_dir)
        self.save_finished_trigger.emit()


class MainWindow(Ui_MainWindow, BaseWindow):

    matting_forms = []
    thread = []

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.m_flag, self.m_Position = False, None
        self.setAcceptDrops(True)
        self.init()

    def init(self):
        self.hint.setText('MODEL HAS NOT BEEN LOADED.')
        self.scrollArea.verticalScrollBar().setStyleSheet(style.scroll_bar)
        self.on_hint_click()
        self.set_listener()

    @pyqtSlot(bool)
    def load_model_callback(self, succeed):
        if succeed:
            self.hint.setHidden(True)
        else:
            self.hint.setText('MODEL LOAD FAILED!')

    @pyqtSlot()
    def matting_finish_callback(self):
        self.hint.setText(f'({Controller().finished_task_num}/{len(Controller().tasks)}) matting...')
        if Controller().finished_task_num == len(Controller().tasks):
            self.hint.setHidden(True)

    def set_listener(self):
        self.close_button.clicked.connect(self.on_close_button_click)
        self.mini_button.clicked.connect(self.on_mini_button_click)
        self.hint.clicked.connect(self.on_hint_click)
        self.scrollArea.verticalScrollBar().rangeChanged.connect(self.on_scroll_range_changed)
        self.open_button.clicked.connect(self.on_open_button_click)
        self.save_button.clicked.connect(self.on_save_button_click)

    @pyqtSlot()
    def on_save_finish_callback(self):
        self.hint.setHidden(True)

    @pyqtSlot()
    def on_save_button_click(self):
        save_dir = QFileDialog.getExistingDirectory(self, 'choose save dur', '/')
        if save_dir == '':
            return
        self.thread = SaveResultsThread(save_dir)
        self.thread.save_finished_trigger.connect(self.on_save_finish_callback)
        self.thread.start()
        self.hint.setText('SAVING...')
        self.hint.setHidden(False)

    @pyqtSlot()
    def on_open_button_click(self):
        img_urls, _ = QFileDialog.getOpenFileNames(self, 'choose images', '/', 'image (*.png;*.jpg;*.jpeg;*.jpe;*.png)')
        if len(img_urls) > 0:
            self.matting(img_urls)

    @pyqtSlot()
    def on_scroll_range_changed(self):
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

    @pyqtSlot()
    def on_hint_click(self):
        if Controller().model_state is not ModelState.LOADED:
            self.hint.setText('LOADING MODEL...')
            self.thread = LoadModelThread()
            self.thread.load_finished_trigger.connect(self.load_model_callback)
            self.thread.start()

    def dragEnterEvent(self, evn):
        evn.accept()

    def dropEvent(self, evn: QtGui.QDropEvent):
        file_urls = evn.mimeData().text().strip().split('\n')
        # image file(.jpg, .jpeg, .png) only.
        img_urls = []
        for url in file_urls:
            if url[:7] != 'file://':
                continue
            if url.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                continue

            img_urls.append(url[8:] if url[9] == ':' else url[7:])  # macOS & windows

        if len(img_urls) > 0:
            self.matting(img_urls)

    def matting(self, img_urls):
        if Controller().model_state != ModelState.LOADED:
            return
        # create tasks and matting forms.
        tasks = Controller().add_tasks(img_urls)
        callbacks = []
        for t in tasks:
            form = MattingForm(t, self.matting_finish_callback)
            callback = form.result_callback
            Controller().add_task_callback(callback)
            callbacks.append(callback)
            self.content.layout().addWidget(form)
            self.matting_forms.append(form)
            # after draw on the window.
            form.matting()

        if Controller().finished_task_num < len(Controller().tasks):
            self.hint.setText(f'({Controller().finished_task_num}/{len(Controller().tasks)}) matting...')
            self.hint.setHidden(False)
