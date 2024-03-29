# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'matting_form.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MattingForm(object):
    def setupUi(self, MattingForm):
        MattingForm.setObjectName("MattingForm")
        MattingForm.resize(812, 709)
        MattingForm.setStyleSheet("QWidget#border_widget{\n"
"background-color: rgb(255, 255, 255);\n"
"border-radius:6px;\n"
"border:2px solid black;\n"
"}")
        self.verticalLayout = QtWidgets.QVBoxLayout(MattingForm)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.border_widget = QtWidgets.QWidget(MattingForm)
        self.border_widget.setStyleSheet("")
        self.border_widget.setObjectName("border_widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.border_widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.widget_4 = QtWidgets.QWidget(self.border_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.img_name = QtWidgets.QLabel(self.widget_4)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.img_name.setFont(font)
        self.img_name.setObjectName("img_name")
        self.horizontalLayout.addWidget(self.img_name)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.detail_button = QtWidgets.QPushButton(self.widget_4)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.detail_button.setFont(font)
        self.detail_button.setStyleSheet("QPushButton{\n"
"background:#F7D674;\n"
"border-radius:6px;\n"
"padding-top:6px;\n"
"padding-bottom:6px;\n"
"padding-left:12px;\n"
"padding-right:12px;\n"
"border:2px solid black;\n"
"}\n"
"QPushButton:hover{\n"
"background:yellow;\n"
"}")
        self.detail_button.setObjectName("detail_button")
        self.horizontalLayout.addWidget(self.detail_button)
        self.change_trimap_button = QtWidgets.QPushButton(self.widget_4)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.change_trimap_button.setFont(font)
        self.change_trimap_button.setStyleSheet("QPushButton{\n"
"background:#F7D674;\n"
"border-radius:6px;\n"
"padding-top:6px;\n"
"padding-bottom:6px;\n"
"padding-left:12px;\n"
"padding-right:12px;\n"
"border:2px solid black;\n"
"}\n"
"QPushButton:hover{\n"
"background:yellow;\n"
"}")
        self.change_trimap_button.setObjectName("change_trimap_button")
        self.horizontalLayout.addWidget(self.change_trimap_button)
        self.change_bg_button = QtWidgets.QPushButton(self.widget_4)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.change_bg_button.setFont(font)
        self.change_bg_button.setStyleSheet("QPushButton{\n"
"background:#F7D674;\n"
"border-radius:6px;\n"
"padding-top:6px;\n"
"padding-bottom:6px;\n"
"padding-left:12px;\n"
"padding-right:12px;\n"
"border:2px solid black;\n"
"}\n"
"QPushButton:hover{\n"
"background:yellow;\n"
"}")
        self.change_bg_button.setObjectName("change_bg_button")
        self.horizontalLayout.addWidget(self.change_bg_button)
        self.verticalLayout_4.addWidget(self.widget_4)
        self.widget_5 = QtWidgets.QWidget(self.border_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_3.setContentsMargins(12, 0, 12, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_2 = QtWidgets.QWidget(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.image_label = QtWidgets.QLabel(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QtCore.QSize(376, 376))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.image_label.setFont(font)
        self.image_label.setStyleSheet("QLabel{\n"
"border:2px dashed darkGray;\n"
"border-radius:10px;\n"
"}")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")
        self.verticalLayout_2.addWidget(self.image_label)
        self.label = QtWidgets.QLabel(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.horizontalLayout_3.addWidget(self.widget_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.widget_3 = QtWidgets.QWidget(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.result_label = QtWidgets.QLabel(self.widget_3)
        self.result_label.setMinimumSize(QtCore.QSize(376, 376))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.result_label.setFont(font)
        self.result_label.setStyleSheet("QLabel{\n"
"border:2px dashed darkGray;\n"
"border-radius:10px;\n"
"}")
        self.result_label.setTextFormat(QtCore.Qt.AutoText)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setObjectName("result_label")
        self.verticalLayout_3.addWidget(self.result_label)
        self.label_2 = QtWidgets.QLabel(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.horizontalLayout_3.addWidget(self.widget_3)
        self.verticalLayout_4.addWidget(self.widget_5)
        self.verticalLayout.addWidget(self.border_widget)

        self.retranslateUi(MattingForm)
        QtCore.QMetaObject.connectSlotsByName(MattingForm)

    def retranslateUi(self, MattingForm):
        _translate = QtCore.QCoreApplication.translate
        MattingForm.setWindowTitle(_translate("MattingForm", "Form"))
        self.img_name.setText(_translate("MattingForm", "TextLabel"))
        self.detail_button.setText(_translate("MattingForm", "V"))
        self.change_trimap_button.setText(_translate("MattingForm", "T"))
        self.change_bg_button.setText(_translate("MattingForm", "B"))
        self.image_label.setText(_translate("MattingForm", "DROP IMAGE HERE"))
        self.label.setText(_translate("MattingForm", "ORIGINAL"))
        self.result_label.setText(_translate("MattingForm", "MATTING..."))
        self.label_2.setText(_translate("MattingForm", "RESULT"))
