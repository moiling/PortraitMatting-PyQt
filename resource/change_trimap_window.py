# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'change_trimap_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ChangeTrimapWindow(object):
    def setupUi(self, ChangeTrimapWindow):
        ChangeTrimapWindow.setObjectName("ChangeTrimapWindow")
        ChangeTrimapWindow.resize(580, 506)
        ChangeTrimapWindow.setStyleSheet("#centralwidget {\n"
"    background-color: rgb(255, 255, 255);\n"
"    border:1px solid darkGray;\n"
"     border-radius:10px;\n"
"      /*border-bottom-right-radius:10px;*/\n"
"}")
        self.centralwidget = QtWidgets.QWidget(ChangeTrimapWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_8 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_8.sizePolicy().hasHeightForWidth())
        self.widget_8.setSizePolicy(sizePolicy)
        self.widget_8.setToolTipDuration(-1)
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_10.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setSpacing(8)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.close_button = QtWidgets.QPushButton(self.widget_8)
        self.close_button.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close_button.sizePolicy().hasHeightForWidth())
        self.close_button.setSizePolicy(sizePolicy)
        self.close_button.setMinimumSize(QtCore.QSize(0, 0))
        self.close_button.setMaximumSize(QtCore.QSize(12, 12))
        self.close_button.setStyleSheet("QPushButton{background:#F76677;border-radius:6px;}QPushButton:hover{background:red;}")
        self.close_button.setText("")
        self.close_button.setIconSize(QtCore.QSize(0, 0))
        self.close_button.setCheckable(False)
        self.close_button.setObjectName("close_button")
        self.horizontalLayout_10.addWidget(self.close_button)
        self.mini_button = QtWidgets.QPushButton(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mini_button.sizePolicy().hasHeightForWidth())
        self.mini_button.setSizePolicy(sizePolicy)
        self.mini_button.setMaximumSize(QtCore.QSize(12, 12))
        self.mini_button.setStyleSheet("QPushButton{background:#F7D674;border-radius:6px;}QPushButton:hover{background:yellow;}")
        self.mini_button.setText("")
        self.mini_button.setIconSize(QtCore.QSize(0, 0))
        self.mini_button.setObjectName("mini_button")
        self.horizontalLayout_10.addWidget(self.mini_button)
        self.verticalLayout.addWidget(self.widget_8)
        self.widget_6 = QtWidgets.QWidget(self.centralwidget)
        self.widget_6.setObjectName("widget_6")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.widget_7 = QtWidgets.QWidget(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.img_name = QtWidgets.QLabel(self.widget_7)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.img_name.setFont(font)
        self.img_name.setObjectName("img_name")
        self.horizontalLayout_8.addWidget(self.img_name)
        self.verticalLayout_4.addWidget(self.widget_7)
        self.widget_4 = QtWidgets.QWidget(self.widget_6)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.trimap_widget = QtWidgets.QWidget(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trimap_widget.sizePolicy().hasHeightForWidth())
        self.trimap_widget.setSizePolicy(sizePolicy)
        self.trimap_widget.setObjectName("trimap_widget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.trimap_widget)
        self.verticalLayout_5.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.image_label = QtWidgets.QLabel(self.trimap_widget)
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
        self.verticalLayout_5.addWidget(self.image_label)
        self.horizontalLayout.addWidget(self.trimap_widget)
        self.widget = QtWidgets.QWidget(self.widget_4)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_5 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy)
        self.widget_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget_5.setStyleSheet("border:2px dashed darkGray;\n"
"border-radius:10px;")
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.f_button = QtWidgets.QPushButton(self.widget_5)
        self.f_button.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.f_button.sizePolicy().hasHeightForWidth())
        self.f_button.setSizePolicy(sizePolicy)
        self.f_button.setMinimumSize(QtCore.QSize(26, 26))
        self.f_button.setMaximumSize(QtCore.QSize(26, 26))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.f_button.setFont(font)
        self.f_button.setStyleSheet("QPushButton{background:#F76677;border-radius:6px;border:2px solid black;}QPushButton:hover{background:rgb(226, 93, 111);}")
        self.f_button.setIconSize(QtCore.QSize(0, 0))
        self.f_button.setCheckable(False)
        self.f_button.setObjectName("f_button")
        self.horizontalLayout_9.addWidget(self.f_button)
        self.b_button = QtWidgets.QPushButton(self.widget_5)
        self.b_button.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_button.sizePolicy().hasHeightForWidth())
        self.b_button.setSizePolicy(sizePolicy)
        self.b_button.setMinimumSize(QtCore.QSize(26, 26))
        self.b_button.setMaximumSize(QtCore.QSize(26, 26))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.b_button.setFont(font)
        self.b_button.setStyleSheet("QPushButton{background:rgb(85, 170, 255);border-radius:6px;border:2px solid black;}QPushButton:hover{background:rgb(77, 155, 232);}")
        self.b_button.setIconSize(QtCore.QSize(0, 0))
        self.b_button.setCheckable(False)
        self.b_button.setObjectName("b_button")
        self.horizontalLayout_9.addWidget(self.b_button)
        self.e_button = QtWidgets.QPushButton(self.widget_5)
        self.e_button.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.e_button.sizePolicy().hasHeightForWidth())
        self.e_button.setSizePolicy(sizePolicy)
        self.e_button.setMinimumSize(QtCore.QSize(26, 26))
        self.e_button.setMaximumSize(QtCore.QSize(26, 26))
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.e_button.setFont(font)
        self.e_button.setStyleSheet("QPushButton{\n"
"background:rgb(255, 255, 255);\n"
"border-radius:6px;\n"
"border:2px solid black;\n"
"}\n"
"QPushButton:hover{background:rgb(243, 243, 243);}")
        self.e_button.setIconSize(QtCore.QSize(0, 0))
        self.e_button.setCheckable(False)
        self.e_button.setObjectName("e_button")
        self.horizontalLayout_9.addWidget(self.e_button)
        self.verticalLayout_2.addWidget(self.widget_5)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.frame_8 = QtWidgets.QFrame(self.widget)
        self.frame_8.setStyleSheet("#frame_8{\n"
"border: 2px dashed darkGray;\n"
"border-radius:6px;\n"
"}")
        self.frame_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_15.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout_15.setSpacing(2)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.frame_12 = QtWidgets.QFrame(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy)
        self.frame_12.setMinimumSize(QtCore.QSize(30, 30))
        self.frame_12.setMaximumSize(QtCore.QSize(30, 30))
        self.frame_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.small_button = QtWidgets.QPushButton(self.frame_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.small_button.sizePolicy().hasHeightForWidth())
        self.small_button.setSizePolicy(sizePolicy)
        self.small_button.setMinimumSize(QtCore.QSize(10, 10))
        self.small_button.setMaximumSize(QtCore.QSize(10, 10))
        self.small_button.setStyleSheet("QPushButton{\n"
"background:white;\n"
"border:2px solid black;\n"
"border-radius:5px;\n"
"}\n"
"QPushButton:hover{background:grey;}")
        self.small_button.setText("")
        self.small_button.setObjectName("small_button")
        self.horizontalLayout_16.addWidget(self.small_button)
        self.horizontalLayout_15.addWidget(self.frame_12)
        self.frame_13 = QtWidgets.QFrame(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy)
        self.frame_13.setMinimumSize(QtCore.QSize(30, 30))
        self.frame_13.setMaximumSize(QtCore.QSize(30, 30))
        self.frame_13.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame_13)
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.middle_button = QtWidgets.QPushButton(self.frame_13)
        self.middle_button.setMinimumSize(QtCore.QSize(18, 18))
        self.middle_button.setMaximumSize(QtCore.QSize(18, 18))
        self.middle_button.setStyleSheet("QPushButton{\n"
"background:white;\n"
"border:2px solid black;\n"
"border-radius:9px;\n"
"}\n"
"QPushButton:hover{background:grey;}")
        self.middle_button.setText("")
        self.middle_button.setObjectName("middle_button")
        self.horizontalLayout_17.addWidget(self.middle_button)
        self.horizontalLayout_15.addWidget(self.frame_13)
        self.frame_14 = QtWidgets.QFrame(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy)
        self.frame_14.setMinimumSize(QtCore.QSize(30, 30))
        self.frame_14.setMaximumSize(QtCore.QSize(30, 30))
        self.frame_14.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.frame_14)
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.large_button = QtWidgets.QPushButton(self.frame_14)
        self.large_button.setMinimumSize(QtCore.QSize(26, 26))
        self.large_button.setMaximumSize(QtCore.QSize(26, 26))
        self.large_button.setStyleSheet("QPushButton{\n"
"background:white;\n"
"border:2px solid black;\n"
"border-radius:13px;\n"
"}\n"
"QPushButton:hover{background:grey;}")
        self.large_button.setText("")
        self.large_button.setObjectName("large_button")
        self.horizontalLayout_18.addWidget(self.large_button)
        self.horizontalLayout_15.addWidget(self.frame_14)
        self.verticalLayout_2.addWidget(self.frame_8)
        self.do_button = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.do_button.setFont(font)
        self.do_button.setStyleSheet("QPushButton{\n"
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
        self.do_button.setObjectName("do_button")
        self.verticalLayout_2.addWidget(self.do_button)
        self.clear_button = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.clear_button.setFont(font)
        self.clear_button.setStyleSheet("QPushButton{\n"
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
        self.clear_button.setObjectName("clear_button")
        self.verticalLayout_2.addWidget(self.clear_button)
        self.ok_button = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setFamily("Terminal")
        font.setPointSize(12)
        self.ok_button.setFont(font)
        self.ok_button.setStyleSheet("QPushButton{\n"
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
        self.ok_button.setObjectName("ok_button")
        self.verticalLayout_2.addWidget(self.ok_button)
        self.horizontalLayout.addWidget(self.widget)
        self.verticalLayout_4.addWidget(self.widget_4)
        self.verticalLayout.addWidget(self.widget_6)
        ChangeTrimapWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(ChangeTrimapWindow)
        QtCore.QMetaObject.connectSlotsByName(ChangeTrimapWindow)

    def retranslateUi(self, ChangeTrimapWindow):
        _translate = QtCore.QCoreApplication.translate
        ChangeTrimapWindow.setWindowTitle(_translate("ChangeTrimapWindow", "Change Trimap"))
        self.img_name.setText(_translate("ChangeTrimapWindow", "TextLabel"))
        self.image_label.setText(_translate("ChangeTrimapWindow", "Trimap"))
        self.f_button.setText(_translate("ChangeTrimapWindow", "F"))
        self.b_button.setText(_translate("ChangeTrimapWindow", "B"))
        self.e_button.setText(_translate("ChangeTrimapWindow", "E"))
        self.do_button.setText(_translate("ChangeTrimapWindow", "DO"))
        self.clear_button.setText(_translate("ChangeTrimapWindow", "CLEAR"))
        self.ok_button.setText(_translate("ChangeTrimapWindow", "OK"))
