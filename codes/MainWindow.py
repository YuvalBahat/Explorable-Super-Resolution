# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets
import numpy as np

MAX_SVD_LAMBDA = 1.
DISPLAY_ESRGAN_RESULTS = True
DEFAULT_BUTTON_SIZE = 30

def ReturnSizePolicy(policy,hasHeightForWidth):
    sizePolicy = QtWidgets.QSizePolicy(policy, policy)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(hasHeightForWidth)
    return sizePolicy


class Ui_MainWindow(object):

    def Define_Grid_layout(self, layout_name, parent, buttons_list, width_height_ratio, layout_cols=None):
        num_buttons = len(buttons_list)
        if layout_cols is None:
            layout_cols = np.maximum(1, int(np.round(np.sqrt(num_buttons / width_height_ratio))))
        layout_rows = int(np.ceil(num_buttons/layout_cols))
        buttons_list += [None for i in range(layout_rows*layout_cols-len(buttons_list))]
        buttons_list = np.reshape(buttons_list, [layout_rows, layout_cols, 3])
        cum_locations = 1*buttons_list[:, :, 1:]
        cum_locations[:,:,0] = np.maximum(cum_locations[:,:,0],np.max(cum_locations[:,:,0],1,keepdims=True))
        cum_locations[:,:,1] = np.maximum(cum_locations[:,:,1],np.max(cum_locations[:,:,1],0,keepdims=True))
        cum_locations = np.stack([np.concatenate([np.zeros([1, layout_cols]).astype(int), np.cumsum(cum_locations[:-1, :, 0], 0)], 0),
                                  np.concatenate([np.zeros([layout_rows, 1]).astype(int), np.cumsum(cum_locations[:, :-1, 1], 1)], 1)], -1)
        widget = QtWidgets.QWidget(self)
        title = QLabel(parent=widget)
        title.setText(layout_name.replace('_',' '))
        setattr(self, layout_name, QtWidgets.QGridLayout(widget))
        new_layout = getattr(self, layout_name)
        new_layout.setContentsMargins(0, title.height(), 0 ,0)
        new_layout.setSpacing(15)
        new_layout.setObjectName(layout_name)
        parent.addWidget(widget)
        for r in range(buttons_list.shape[0]):
            for c in range(buttons_list.shape[1]):
                if (r+1)*(c+1)>num_buttons:
                    break
                new_layout.addWidget(buttons_list[r,c,0],  cum_locations[r,c,0],cum_locations[r,c,1], buttons_list[r,c,1], buttons_list[r,c,2])

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(10,10)

        # Configuring parameters:
        self.max_SVD_Lambda = MAX_SVD_LAMBDA
        self.display_ESRGAN = DISPLAY_ESRGAN_RESULTS
        self.button_size = DEFAULT_BUTTON_SIZE

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setSizePolicy(ReturnSizePolicy(QtWidgets.QSizePolicy.Maximum,self.centralWidget.sizePolicy().hasHeightForWidth()))
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setSizePolicy(ReturnSizePolicy(QtWidgets.QSizePolicy.Maximum,self.widget.sizePolicy().hasHeightForWidth()))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(15)
        self.gridLayout.setObjectName("gridLayout")

        self.USE_LAYOUTS_METHOD = True
        if not self.USE_LAYOUTS_METHOD:
            self.ZToolbar_widget = QtWidgets.QWidget(MainWindow)
            self.ZToolbar = QtWidgets.QGridLayout(self.ZToolbar_widget)
            self.ZToolbar.setContentsMargins(11, 11, 11, 11)
            self.ZToolbar.setSpacing(15)
            self.ZToolbar.setObjectName("ZToolbar")
            self.verticalLayout_2.addWidget(self.ZToolbar_widget)

        self.canvas.sliderZ0 = QSlider()
        self.canvas.sliderZ0.setObjectName('sliderZ0')
        self.canvas.sliderZ0.setRange(0, 100*self.max_SVD_Lambda)
        self.canvas.sliderZ0.setSliderPosition(100*self.max_SVD_Lambda/2)
        self.canvas.sliderZ0.setSingleStep(1)
        self.canvas.sliderZ0.setOrientation(Qt.Vertical)
        self.canvas.sliderZ0.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=0,dont_update_undo_list=True))
        self.canvas.sliderZ0.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.sliderZ0.value() / 100, index=0))
        self.canvas.sliderZ0.setToolTip('Primary direction gradients magnitude')
        if not self.USE_LAYOUTS_METHOD:
            self.ZToolbar.addWidget(self.canvas.sliderZ0,1,0,4,1)
        self.canvas.sliderZ1 = QSlider()
        self.canvas.sliderZ1.setObjectName('sliderZ1')
        self.canvas.sliderZ1.setRange(0, 100*self.max_SVD_Lambda)
        self.canvas.sliderZ1.setSliderPosition(100*self.max_SVD_Lambda/2)
        self.canvas.sliderZ1.setSingleStep(1)
        self.canvas.sliderZ1.setOrientation(Qt.Vertical)
        self.canvas.sliderZ1.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=1,dont_update_undo_list=True))
        self.canvas.sliderZ1.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.sliderZ1.value() / 100, index=1))
        self.canvas.sliderZ1.setToolTip('Secondary direction gradients magnitude')
        if not self.USE_LAYOUTS_METHOD:
            self.ZToolbar.addWidget(self.canvas.sliderZ1,1,1,4,1)
        self.canvas.slider_third_channel = QDial()
        self.canvas.slider_third_channel.setWrapping(True)
        self.canvas.slider_third_channel.setNotchesVisible(True)
        self.canvas.slider_third_channel.setObjectName('slider_third_channel')
        self.canvas.slider_third_channel.setRange(-100*np.pi, 100*np.pi)
        self.canvas.slider_third_channel.setSingleStep(1)
        self.canvas.slider_third_channel.setOrientation(Qt.Vertical)
        self.canvas.slider_third_channel.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=2,dont_update_undo_list=True))
        self.canvas.slider_third_channel.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.slider_third_channel.value() / 100, index=2))
        if not self.USE_LAYOUTS_METHOD:
            self.ZToolbar.addWidget(self.canvas.slider_third_channel,1,2,4,4)

        self.DisplayedImageSelection_button = QtWidgets.QComboBox(MainWindow)
        self.DisplayedImageSelection_button.setObjectName("DisplayedImageSelection_button")
        self.DisplayedImageSelection_button.setEnabled(False)
        self.DisplayedImageSelection_button.setToolTip('Displayed image')
        self.DisplayedImageSelection_button.highlighted.connect(self.SelectImage2Display)
        if self.display_ESRGAN:
            self.DisplayedImageSelection_button.addItem('ESRGAN')
            self.DisplayedImageSelection_button.setEnabled(True)
            self.ESRGAN_index = self.DisplayedImageSelection_button.findText('ESRGAN')
        if True: #LOAD_HR_IMAGE: Now I always add this display, and only enable it for images with GT
            self.DisplayedImageSelection_button.addItem('GT')
            self.DisplayedImageSelection_button.setEnabled(True)
            self.GT_HR_index = self.DisplayedImageSelection_button.findText('GT')
        else:
            self.GT_HR_index = None
        self.DisplayedImageSelection_button.addItem('Z')
        self.cur_Z_im_index = self.DisplayedImageSelection_button.findText('Z')
        self.canvas.cur_Z_im_index = self.cur_Z_im_index
        self.canvas.current_display_index = 1*self.cur_Z_im_index
        self.DisplayedImageSelection_button.addItems([str(i+1) for i in range(self.num_random_Zs)])
        self.random_display_indexes = [self.DisplayedImageSelection_button.findText(str(i+1)) for i in range(self.num_random_Zs)]
        self.DisplayedImageSelection_button.addItem('Scribble')
        self.canvas.scribble_display_index = self.DisplayedImageSelection_button.findText('Scribble')

        self.randomLimitingWeightBox_Enabled = False
        if self.randomLimitingWeightBox_Enabled:
            self.randomLimitingWeightBox = QtWidgets.QDoubleSpinBox(MainWindow)
            self.randomLimitingWeightBox.setObjectName("randomLimitingWeightBox")
            self.randomLimitingWeightBox.setValue(1.)
            self.randomLimitingWeightBox.setMaximum(200)
            self.randomLimitingWeightBox.setToolTip('Random images limitation weight')

        self.periodicity_mag_1 = QtWidgets.QDoubleSpinBox(MainWindow)
        self.periodicity_mag_1.setObjectName("periodicity_mag_1")
        self.periodicity_mag_1.setValue(1.)
        self.periodicity_mag_1.setMaximum(200)
        self.periodicity_mag_1.setSingleStep(0.1)
        self.periodicity_mag_1.setDecimals(1)
        self.periodicity_mag_1.setToolTip('Primary period length')
        self.periodicity_mag_2 = QtWidgets.QDoubleSpinBox(MainWindow)
        self.periodicity_mag_2.setObjectName("periodicity_mag_2")
        self.periodicity_mag_2.setValue(1.)
        self.periodicity_mag_2.setMaximum(200)
        self.periodicity_mag_2.setSingleStep(0.1)
        self.periodicity_mag_2.setDecimals(1)
        self.periodicity_mag_2.setToolTip('Secondary period length')

        def Define_Push_Button(button_name,tooltip,disabled=False,checkable=False,size=1):
            setattr(self,button_name+'_button',QPushButton(icon=QIcon(QPixmap('icons/'+button_name+'.png'))))
            button = getattr(self,button_name+'_button')
            button.setObjectName(button_name+'_button')
            button.setEnabled(not disabled)
            button.setToolTip(tooltip)
            button.setCheckable(checkable)
            if not isinstance(size,list):
                size = [size,size]
            button.setMinimumSize(QtCore.QSize(size[0]*self.button_size,size[1]*self.button_size))
            button.setMaximumSize(QtCore.QSize(size[0]*self.button_size,size[1]*self.button_size))

        Define_Push_Button(button_name='CopyFromRandom',tooltip='Copy displayed random region to Z',disabled=True)
        Define_Push_Button(button_name='Copy2Random', tooltip='Copy region from Z to random images')
        Define_Push_Button(button_name='indicatePeriodicity', tooltip='Set desired periodicity', checkable=True)
        if self.USE_LAYOUTS_METHOD:
            self.Define_Grid_layout('ZToolbar', parent=self.verticalLayout_2,
                                    buttons_list=[(self.canvas.sliderZ0,4,1), (self.canvas.sliderZ1,4,1),
                                                  (self.canvas.slider_third_channel,4,4),
                                                  (self.DisplayedImageSelection_button,1,4), (self.CopyFromRandom_button, 1, 1),
                                                  (self.Copy2Random_button, 1, 1), (self.indicatePeriodicity_button, 1, 1),
                                                  (self.periodicity_mag_1,1,4), (self.periodicity_mag_2,1,4)],
                                    width_height_ratio=1)

        Define_Push_Button('selectrect', tooltip='Rectangle selection', checkable=True)
        self.gridLayout.addWidget(self.selectrect_button, 0, 1, 1, 1)

        Define_Push_Button('selectpoly', tooltip='Polygon selection', checkable=True)
        self.gridLayout.addWidget(self.selectpoly_button, 0, 0, 1, 1)

        Define_Push_Button('unselect', tooltip='De-select', disabled=False, checkable=False)
        self.gridLayout.addWidget(self.unselect_button,0,2, 1, 1)

        Define_Push_Button('invertSelection', tooltip='Invert selection', disabled=False, checkable=False)
        self.gridLayout.addWidget(self.invertSelection_button,0,3, 1, 1)

        Define_Push_Button('uniformZ', tooltip='Spatially uniform Z', disabled=False, checkable=False)
        self.gridLayout.addWidget(self.uniformZ_button,2,1, 1, 1)

        Define_Push_Button('desiredExternalAppearanceMode', tooltip='Select external desired region', disabled=False, checkable=True)
        self.gridLayout.addWidget(self.desiredExternalAppearanceMode_button, 1, 0, 1, 1)

        Define_Push_Button('desiredAppearanceMode', tooltip='Select desired region within image', disabled=False, checkable=True)
        self.gridLayout.addWidget(self.desiredAppearanceMode_button, 1,1, 1, 1)

        Define_Push_Button('Zdisplay', tooltip='Toggle Z display', disabled=False, checkable=True)
        self.gridLayout.addWidget(self.Zdisplay_button, 2,0, 1, 1)

        Define_Push_Button('undo_scribble', tooltip='Undo scribble/imprint', disabled=True)
        self.gridLayout.addWidget(self.undo_scribble_button, 3, 2, 1, 1)

        Define_Push_Button('redo_scribble', tooltip='Redo scribble/imprint', disabled=True)
        self.gridLayout.addWidget(self.redo_scribble_button, 3, 3, 1, 1)

        # Imprinting translation buttons:
        for button_num,button in enumerate(['left','right','up','down']):
            Define_Push_Button(button+'_imprinting', tooltip='Move imprinting '+button, disabled=True)
            self.gridLayout.addWidget(getattr(self,button+'_imprinting_button'), 4, button_num, 1, 1)
        # Imprinting dimensions change buttons:
        for button_num,button in enumerate(['narrower','wider','taller','shorter']):
            Define_Push_Button(button+'_imprinting', tooltip='Make imprinting '+button, disabled=True)
            self.gridLayout.addWidget(getattr(self,'%s_imprinting_button'%(button)), 5, button_num, 1, 1)

        Define_Push_Button('undoZ', tooltip='Undo image manipulation', disabled=True)
        self.gridLayout.addWidget(self.undoZ_button, 3, 0, 1, 1)

        Define_Push_Button('redoZ', tooltip='Redo image manipulation', disabled=True)
        self.gridLayout.addWidget(self.redoZ_button, 3, 1, 1, 1)

        self.auto_hist_temperature_mode_Enabled = False
        if self.auto_hist_temperature_mode_Enabled:
            Define_Push_Button('auto_hist_temperature_mode', tooltip='Automatic histogram temperature', checkable=True)
            self.gridLayout.addWidget(self.auto_hist_temperature_mode_button, 2, 2, 1, 1)

        self.verticalLayout_2.addWidget(self.widget)
        # spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.phisical_canvas = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phisical_canvas.sizePolicy().hasHeightForWidth())
        self.phisical_canvas.setSizePolicy(sizePolicy)
        self.phisical_canvas.setText("")
        self.phisical_canvas.setObjectName("phisical_canvas")
        self.horizontalLayout.addWidget(self.phisical_canvas)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        # spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 549, 22))
        self.menuBar.setObjectName("menuBar")
        self.menuFIle = QtWidgets.QMenu(self.menuBar)
        self.menuFIle.setObjectName("menuFIle")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuImage = QtWidgets.QMenu(self.menuBar)
        self.menuImage.setObjectName("menuImage")
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.fileToolbar = QtWidgets.QToolBar(MainWindow)
        self.fileToolbar.setIconSize(QtCore.QSize(16, 16))
        self.fileToolbar.setObjectName("fileToolbar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.fileToolbar)

        MainWindow.addToolBarBreak()#.addItem(spacerItem2)

        self.Scribble_Toolbar = QtWidgets.QToolBar(MainWindow)
        self.Scribble_Toolbar.setIconSize(QtCore.QSize(25,25))
        self.Scribble_Toolbar.setObjectName("Scribble_Toolbar")
        self.Scribble_Toolbar.setOrientation(QtCore.Qt.Horizontal)
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.Scribble_Toolbar)

        self.ZToolbar2 = QtWidgets.QToolBar(MainWindow)
        self.ZToolbar2.setIconSize(QtCore.QSize(25,25))
        self.ZToolbar2.setObjectName("ZToolbar2")
        self.ZToolbar2.setOrientation(QtCore.Qt.Horizontal)
        # MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea if MainWindow.canvas.display_zoom_factor>1 else QtCore.Qt.BottomToolBarArea, self.ZToolbar2)
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.ZToolbar2)

        Define_Push_Button(button_name='scribble_reset',tooltip='Erase scribble in region')
        # icon_scribble_reset = QIcon()
        # icon_scribble_reset.addPixmap(QPixmap("icons/scribble_reset.png"), QIcon.Normal, QIcon.Off)
        # self.actionScribbleReset = QtWidgets.QAction(icon=icon_scribble_reset,parent=self.Scribble_Toolbar)
        # self.actionScribbleReset.setObjectName("actionScribbleReset")
        # self.actionScribbleReset.setEnabled(True)

        Define_Push_Button(button_name='apply_scribble',tooltip='Perform a single scribble/imprinting application step',disabled=True)
        # icon_scribble_reset = QIcon()
        # icon_scribble_reset.addPixmap(QPixmap("icons/apply_scribble.png"), QIcon.Normal, QIcon.Off)
        # self.actionApplyScrible = QtWidgets.QAction(icon=icon_scribble_reset,parent=self.Scribble_Toolbar)
        # self.actionApplyScrible.setObjectName("actionApplyScrible")

        Define_Push_Button(button_name='loop_apply_scribble',tooltip='Perform multiple scribble/imprinting application steps',disabled=True)
        # icon_scribble_reset = QIcon()
        # icon_scribble_reset.addPixmap(QPixmap("icons/loop_apply_scribble.png"), QIcon.Normal, QIcon.Off)
        # self.actionLoopApplyScrible = QtWidgets.QAction(icon=icon_scribble_reset,parent=self.Scribble_Toolbar)
        # self.actionLoopApplyScrible.setObjectName("actionLoopApplyScrible")

        Define_Push_Button(button_name='pencil',tooltip='Pencil',checkable=True)
        # icon8 = QIcon()
        # icon8.addPixmap(QPixmap("icons/pencil.png"), QIcon.Normal, QIcon.Off)
        # self.pen_button = QtWidgets.QPushButton(icon=icon8,parent=self.Scribble_Toolbar)
        # self.pen_button.setIcon(icon8)
        # self.pen_button.setCheckable(True)
        # self.pen_button.setObjectName("pen_button")
        # self.pen_button.setToolTip('Pen')

        Define_Push_Button(button_name='dropper',tooltip='Eyedropper',checkable=True)
        # icon_dropper = QIcon()
        # icon_dropper.addPixmap(QPixmap("icons/pipette.png"), QIcon.Normal, QIcon.Off)
        # self.dropper_button = QtWidgets.QPushButton(parent=self.widget,icon=icon_dropper)
        # self.dropper_button.setCheckable(True)
        # self.dropper_button.setObjectName("dropper_button")
        # self.dropper_button.setToolTip('Eyedropper')

        Define_Push_Button(button_name='line',tooltip='Straight line drawing',checkable=True)
        # line_icon = QIcon()
        # line_icon.addPixmap(QPixmap("icons/layer-shape-line.png"), QIcon.Normal, QIcon.Off)
        # self.line_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=line_icon)
        # self.line_button.setCheckable(True)
        # self.line_button.setObjectName("line_button")
        # self.line_button.setToolTip('Straight line drawing')

        Define_Push_Button(button_name='polygon',tooltip='Polygon drawing',checkable=True)
        # icon11 = QIcon()
        # icon11.addPixmap(QPixmap("icons/layer-shape-polygon.png"), QIcon.Normal, QIcon.Off)
        # self.polygon_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=icon11)
        # # self.polygon_button.setIcon(icon11)
        # self.polygon_button.setCheckable(True)
        # self.polygon_button.setObjectName("polygon_button")
        # self.polygon_button.setToolTip('Polygon drawing')

        Define_Push_Button(button_name='rect',tooltip='Rectangle drawing',checkable=True)
        # icon12 = QIcon()
        # icon12.addPixmap(QPixmap("icons/layer-shape.png"), QIcon.Normal, QIcon.Off)
        # self.rect_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=icon12)
        # self.rect_button.setCheckable(True)
        # self.rect_button.setObjectName("rect_button")
        # self.rect_button.setToolTip('Rectangle drawing')

        Define_Push_Button(button_name='im_input',tooltip='Set imprinting rectangle (Using transparent color)',checkable=True,disabled=True)
        # icon_image_input = QIcon()
        # icon_image_input.addPixmap(QPixmap("icons/image_input.png"), QIcon.Normal, QIcon.Off)
        # self.im_input_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=icon_image_input)
        # self.im_input_button.setCheckable(True)
        # self.im_input_button.setObjectName("im_input_button")
        # self.im_input_button.setEnabled(False)
        # self.im_input_button.setToolTip('Set imprinting rectangle (Using transparent color)')

        Define_Push_Button(button_name='im_input_auto_location',tooltip='Set boundaries for automatic imprinting location (Using transparent color)',checkable=True,disabled=True)
        # icon_im_input_auto_location = QIcon()
        # icon_im_input_auto_location.addPixmap(QPixmap("icons/image_input_noDC.png"), QIcon.Normal, QIcon.Off)
        # self.im_input_auto_location_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=icon_im_input_auto_location)
        # self.im_input_auto_location_button.setCheckable(True)
        # self.im_input_auto_location_button.setObjectName("im_input_auto_location_button")
        # self.im_input_auto_location_button.setEnabled(False)
        # self.im_input_auto_location_button.setToolTip('Set boundaries for automatic imprinting location (Using transparent color)')

        Define_Push_Button(button_name='ellipse',tooltip='Ellipse drawing',checkable=True)
        # icon13 = QIcon()
        # icon13.addPixmap(QPixmap("icons/layer-shape-ellipse.png"), QIcon.Normal, QIcon.Off)
        # self.ellipse_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=icon13)
        # self.ellipse_button.setCheckable(True)
        # self.ellipse_button.setObjectName("ellipse_button")
        # self.ellipse_button.setToolTip('Ellipse drawing')

        # self.actionCopy = QtWidgets.QAction(MainWindow)
        # self.actionCopy.setObjectName("actionCopy")
        # self.actionClearImage = QtWidgets.QAction(MainWindow)
        # self.actionClearImage.setObjectName("actionClearImage")

        Define_Push_Button(button_name='open_image',tooltip='Load LR image')
        # self.actionOpenImage = QtWidgets.QAction(MainWindow)
        # icon16 = QIcon()
        # icon16.addPixmap(QPixmap("icons/blue-folder-open-image.png"), QIcon.Normal, QIcon.Off)
        # self.actionOpenImage.setIcon(icon16)
        # self.actionOpenImage.setObjectName("actionOpenImage")
        # self.actionOpenImage.setToolTip('Load LR image')

        Define_Push_Button(button_name='Z_load',tooltip='Load Z map')
        # self.actionLoad_Z = QtWidgets.QAction(MainWindow)
        # icon_load_Z = QIcon()
        # icon_load_Z.addPixmap(QPixmap("icons/Z_load.png"), QIcon.Normal, QIcon.Off)
        # self.actionLoad_Z.setIcon(icon_load_Z)
        # self.actionLoad_Z.setObjectName("actionLoad_Z")
        # self.actionLoad_Z.setToolTip('Load Z map')

        Define_Push_Button(button_name='Z_mask_load',tooltip='Load Z map to infer selection')
        # self.actionLoad_Z_mask = QtWidgets.QAction(MainWindow)
        # icon_load_Z_mask = QIcon()
        # icon_load_Z_mask.addPixmap(QPixmap("icons/Z_mask_load.png"), QIcon.Normal, QIcon.Off)
        # self.actionLoad_Z_mask.setIcon(icon_load_Z_mask)
        # self.actionLoad_Z_mask.setObjectName("actionLoad_Z_mask")
        # self.actionLoad_Z_mask.setToolTip('Load Z map to infer selection')

        self.estimatedKenrel_button = QtWidgets.QPushButton(self.fileToolbar)
        self.estimatedKenrel_button.setSizePolicy(ReturnSizePolicy(QtWidgets.QSizePolicy.Fixed,self.estimatedKenrel_button.sizePolicy().hasHeightForWidth()))
        self.estimatedKenrel_button.setMinimumSize(QtCore.QSize(30, 30))
        self.estimatedKenrel_button.setMaximumSize(QtCore.QSize(30, 30))
        self.estimatedKenrel_button.setText("")
        icon4 = QIcon()
        icon4.addPixmap(QPixmap("icons/hist.png"), QIcon.Normal, QIcon.Off)
        self.estimatedKenrel_button.setIcon(icon4)
        self.estimatedKenrel_button.setCheckable(True)
        self.estimatedKenrel_button.setObjectName("desiredHistMode_button")
        self.estimatedKenrel_button.setToolTip('Use estimated SR kernel')

        self.actionProcessRandZ = QtWidgets.QAction(MainWindow)
        # self.actionProcessRandZ.setText("Random Z")
        icon_randomZ = QIcon()
        icon_randomZ.addPixmap(QPixmap("icons/random_l1.png"), QIcon.Normal, QIcon.Off)
        self.actionProcessRandZ.setIcon(icon_randomZ)
        self.actionProcessRandZ.setObjectName("actionProcessRandZ")
        self.actionProcessRandZ.setToolTip('Produce random images')

        self.actionProcessLimitedRandZ = QtWidgets.QAction(MainWindow)
        icon_limited_randomZ = QIcon()
        icon_limited_randomZ.addPixmap(QPixmap("icons/random_l1_limited.png"), QIcon.Normal, QIcon.Off)
        self.actionProcessLimitedRandZ.setIcon(icon_limited_randomZ)
        self.actionProcessLimitedRandZ.setObjectName("actionProcessLimitedRandZ")
        self.actionProcessLimitedRandZ.setToolTip('Produce random images close to current')

        # self.randomLimitingWeightBox = QtWidgets.QLineEdit(MainWindow)
        # self.randomLimitingWeightBox_Enabled = False
        # if self.randomLimitingWeightBox_Enabled:
        #     self.randomLimitingWeightBox = QtWidgets.QDoubleSpinBox(MainWindow)
        #     self.randomLimitingWeightBox.setObjectName("randomLimitingWeightBox")
        #     self.randomLimitingWeightBox.setValue(1.)
        #     # self.randomLimitingWeightBox.setText('1')
        #     self.randomLimitingWeightBox.setMaximum(200)
        #     self.randomLimitingWeightBox.setToolTip('Random images limitation weight')
        #     # self.randomLimitingWeightBox.setSingleStep(0.1)
        #
        # self.periodicity_mag_1 = QtWidgets.QDoubleSpinBox(MainWindow)
        # self.periodicity_mag_1.setObjectName("periodicity_mag_1")
        # self.periodicity_mag_1.setValue(1.)
        # self.periodicity_mag_1.setMaximum(200)
        # self.periodicity_mag_1.setSingleStep(0.1)
        # self.periodicity_mag_1.setDecimals(1)
        # self.periodicity_mag_1.setToolTip('Primary period length')
        # self.periodicity_mag_2 = QtWidgets.QDoubleSpinBox(MainWindow)
        # self.periodicity_mag_2.setObjectName("periodicity_mag_2")
        # self.periodicity_mag_2.setValue(1.)
        # self.periodicity_mag_2.setMaximum(200)
        # self.periodicity_mag_2.setSingleStep(0.1)
        # self.periodicity_mag_2.setDecimals(1)
        # self.periodicity_mag_2.setToolTip('Secondary period length')
        #
        # icon_copyRandZ = QIcon()
        # icon_copyRandZ.addPixmap(QPixmap("icons/randoms2default.png"), QIcon.Normal, QIcon.Off)
        # self.actionCopyFromRandom = QtWidgets.QPushButton(icon=icon_copyRandZ,parent=MainWindow)
        # self.actionCopyFromRandom.setObjectName("actionCopyFromRandom")
        # self.actionCopyFromRandom.setEnabled(False)
        # self.actionCopyFromRandom.setToolTip('Copy displayed random region to Z')
        #
        # self.actionCopy2Random = QtWidgets.QPushButton(MainWindow)
        # icon_copyDefault2RandZ = QIcon()
        # icon_copyDefault2RandZ = QIcon()
        # icon_copyDefault2RandZ.addPixmap(QPixmap("icons/default2randoms.png"), QIcon.Normal, QIcon.Off)
        # self.actionCopy2Random.setIcon(icon_copyDefault2RandZ)
        # self.actionCopy2Random.setObjectName("actionCopy2Random")
        # self.actionCopy2Random.setToolTip('Copy region from Z to random images')
        # # self.actionCopy2Random.setEnabled(True)

        patch_opt_behavior_icon = QIcon()
        patch_opt_behavior_icon.addPixmap(QPixmap("icons/layer-shape-ellipse.png"), QIcon.Normal, QIcon.Off)
        self.special_behavior_button = QtWidgets.QPushButton(parent=self.ZToolbar2,icon=patch_opt_behavior_icon)
        self.special_behavior_button.setCheckable(True)
        self.special_behavior_button.setObjectName("special_behavior_button")
        self.special_behavior_button.setToolTip('Toggle special behavior')

        self.actionIncreaseSTD = QtWidgets.QAction(MainWindow)
        # self.actionIncreaseSTD.setText("Increase STD")
        self.actionIncreaseSTD.setObjectName("actionIncreaseSTD")
        icon_sigmaUp = QIcon()
        icon_sigmaUp.addPixmap(QPixmap("icons/sigma_up.png"), QIcon.Normal, QIcon.Off)
        self.actionIncreaseSTD.setIcon(icon_sigmaUp)
        self.actionIncreaseSTD.setToolTip('Increase local STD (Magnitude)')

        self.actionDecreaseSTD = QtWidgets.QAction(MainWindow)
        # self.actionDecreaseSTD.setText("Decrease STD")
        self.actionDecreaseSTD.setObjectName("actionDecreaseSTD")
        icon_sigmadown = QIcon()
        icon_sigmadown.addPixmap(QPixmap("icons/sigma_down.png"), QIcon.Normal, QIcon.Off)
        self.actionDecreaseSTD.setIcon(icon_sigmadown)
        self.actionDecreaseSTD.setToolTip('Decrease local STD (Magnitude)')

        self.STD_increment = QtWidgets.QDoubleSpinBox(MainWindow)
        self.STD_increment.setObjectName("STD_increment")
        self.STD_increment.setValue(0.03)
        self.STD_increment.setRange(0,0.3)
        self.STD_increment.setSingleStep(0.01)
        self.STD_increment.setDecimals(2)
        self.STD_increment.setToolTip('STD/Brightness change')

        self.actionDecreaseTV = QtWidgets.QAction(MainWindow)
        self.actionDecreaseTV.setObjectName("actionDecreaseTV")
        icon_TVdown = QIcon()
        icon_TVdown.addPixmap(QPixmap("icons/TV_down.png"), QIcon.Normal, QIcon.Off)
        self.actionDecreaseTV.setIcon(icon_TVdown)
        self.actionDecreaseTV.setToolTip('Decrease TV')

        self.actionImitateHist = QtWidgets.QAction(MainWindow)
        self.actionImitateHist.setObjectName("actionImitateHist")
        icon_hist_imitation = QIcon()
        icon_hist_imitation.addPixmap(QPixmap("icons/hist_imitation.png"), QIcon.Normal, QIcon.Off)
        self.actionImitateHist.setIcon(icon_hist_imitation)
        self.actionImitateHist.setEnabled(False)
        self.actionImitateHist.setToolTip('Encourage desired histogram')

        self.actionImitatePatchHist = QtWidgets.QAction(MainWindow)
        self.actionImitatePatchHist.setObjectName("actionImitatePatchHist")
        icon_patch_hist_imitation = QIcon()
        icon_patch_hist_imitation.addPixmap(QPixmap("icons/hist_imitation_patch.png"), QIcon.Normal, QIcon.Off)
        self.actionImitatePatchHist.setIcon(icon_patch_hist_imitation)
        self.actionImitatePatchHist.setEnabled(False)
        self.actionImitatePatchHist.setToolTip('Encourage desired mean-less (normalized) patches')

        # self.actionMatchSliders = QtWidgets.QAction(MainWindow)
        # self.actionMatchSliders.setObjectName("actionMatchSliders")
        # icon_match_sliders = QIcon()
        # icon_match_sliders.addPixmap(QPixmap("icons/sliders_icon.png"), QIcon.Normal, QIcon.Off)
        # self.actionMatchSliders.setIcon(icon_match_sliders)

        self.actionFoolAdversary = QtWidgets.QAction(MainWindow)
        self.actionFoolAdversary.setObjectName("actionFoolAdversary")
        icon_pixmap = QPixmap("icons/adversary.png")
        # icon_pixmap.scaledToWidth(40)
        icon_adversary = QIcon()
        icon_adversary.addPixmap(icon_pixmap, QIcon.Normal, QIcon.Off)
        # icon_adversary.actualSize(QtCore.QSize(60, 60))
        self.actionFoolAdversary.setIcon(icon_adversary)
        self.actionFoolAdversary.setToolTip('Fool discriminator')

        self.actionIncreasePeriodicity = QtWidgets.QAction(MainWindow)
        self.actionIncreasePeriodicity.setObjectName("actionIncreasePeriodicity")
        icon_pixmap = QPixmap("icons/indicate_periodicity_2D.png")
        icon_periodicity = QIcon()
        icon_periodicity.addPixmap(icon_pixmap, QIcon.Normal, QIcon.Off)
        self.actionIncreasePeriodicity.setIcon(icon_periodicity)
        self.actionIncreasePeriodicity.setEnabled(False)
        self.actionIncreasePeriodicity.setToolTip('Increase 2D periodicity')

        self.actionIncreasePeriodicity_1D = QtWidgets.QAction(MainWindow)
        self.actionIncreasePeriodicity_1D.setObjectName("actionIncreasePeriodicity_1D")
        icon_pixmap = QPixmap("icons/indicate_periodicity_1D.png")
        icon_periodicity = QIcon()
        icon_periodicity.addPixmap(icon_pixmap, QIcon.Normal, QIcon.Off)
        self.actionIncreasePeriodicity_1D.setIcon(icon_periodicity)
        self.actionIncreasePeriodicity_1D.setEnabled(False)
        self.actionIncreasePeriodicity_1D.setToolTip('Increase 1D periodicity')

        self.actionSaveImageAndData = QtWidgets.QAction(MainWindow)
        icon17_0 = QIcon()
        icon17_0.addPixmap(QPixmap("icons/Save-icon.png"), QIcon.Normal, QIcon.Off)
        self.actionSaveImageAndData.setIcon(icon17_0)
        self.actionSaveImageAndData.setObjectName("actionSaveImageAndData")
        self.actionSaveImageAndData.setToolTip('Save image, Z and scribble data')

        self.actionDecreaseDisplayZoom = QtWidgets.QAction(MainWindow)
        icon17_0 = QIcon()
        icon17_0.addPixmap(QPixmap("icons/zoomout.png"), QIcon.Normal, QIcon.Off)
        self.actionDecreaseDisplayZoom.setIcon(icon17_0)
        self.actionDecreaseDisplayZoom.setObjectName("actionDecreaseDisplayZoom")
        self.actionDecreaseDisplayZoom.setToolTip('Save image, Z and scribble data')

        self.actionIncreaseDisplayZoom = QtWidgets.QAction(MainWindow)
        icon17_0 = QIcon()
        icon17_0.addPixmap(QPixmap("icons/zoomin.png"), QIcon.Normal, QIcon.Off)
        self.actionIncreaseDisplayZoom.setIcon(icon17_0)
        self.actionIncreaseDisplayZoom.setObjectName("actionIncreaseDisplayZoom")
        self.actionIncreaseDisplayZoom.setToolTip('Save image, Z and scribble data')

        self.actionSaveImage = QtWidgets.QAction(MainWindow)
        icon17 = QIcon()
        icon17.addPixmap(QPixmap("icons/disk.png"), QIcon.Normal, QIcon.Off)
        self.actionSaveImage.setIcon(icon17)
        self.actionSaveImage.setObjectName("actionSaveImage")
        self.actionInvertColors = QtWidgets.QAction(MainWindow)
        self.actionInvertColors.setObjectName("actionInvertColors")
        self.actionFlipHorizontal = QtWidgets.QAction(MainWindow)
        self.actionFlipHorizontal.setObjectName("actionFlipHorizontal")
        self.actionFlipVertical = QtWidgets.QAction(MainWindow)
        self.actionFlipVertical.setObjectName("actionFlipVertical")
        self.actionNewImage = QtWidgets.QAction(MainWindow)
        icon18 = QIcon()
        icon18.addPixmap(QPixmap("icons/document-image.png"), QIcon.Normal, QIcon.Off)
        self.actionNewImage.setIcon(icon18)
        self.actionNewImage.setObjectName("actionNewImage")
        self.actionNewImage.setToolTip('Synthetically downscale an HR image')
        self.actionBold = QtWidgets.QAction(MainWindow)
        self.actionBold.setCheckable(True)
        icon19 = QIcon()
        icon19.addPixmap(QPixmap("icons/edit-bold.png"), QIcon.Normal, QIcon.Off)
        self.actionBold.setIcon(icon19)
        self.actionBold.setObjectName("actionBold")
        self.actionItalic = QtWidgets.QAction(MainWindow)
        self.actionItalic.setCheckable(True)
        icon20 = QIcon()
        icon20.addPixmap(QPixmap("icons/edit-italic.png"), QIcon.Normal, QIcon.Off)
        self.actionItalic.setIcon(icon20)
        self.actionItalic.setObjectName("actionItalic")
        self.actionUnderline = QtWidgets.QAction(MainWindow)
        self.actionUnderline.setCheckable(True)
        icon21 = QIcon()
        icon21.addPixmap(QPixmap("icons/edit-underline.png"), QIcon.Normal, QIcon.Off)
        self.actionUnderline.setIcon(icon21)
        self.actionUnderline.setObjectName("actionUnderline")
        self.actionFillShapes = QtWidgets.QAction(MainWindow)
        self.actionFillShapes.setCheckable(True)
        icon22 = QIcon()
        icon22.addPixmap(QPixmap("icons/paint-can-color.png"), QIcon.Normal, QIcon.Off)
        self.actionFillShapes.setIcon(icon22)
        self.actionFillShapes.setObjectName("actionFillShapes")
        self.menuFIle.addAction(self.actionNewImage)
        # self.menuFIle.addAction(self.open_image_button)
        self.menuFIle.addAction(self.actionSaveImage)
        self.menuFIle.addAction(self.actionSaveImageAndData)
        # self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addSeparator()
        # self.menuEdit.addAction(self.actionClearImage)
        self.menuImage.addAction(self.actionInvertColors)
        self.menuImage.addSeparator()
        self.menuImage.addAction(self.actionFlipHorizontal)
        self.menuImage.addAction(self.actionFlipVertical)
        self.menuBar.addAction(self.menuFIle.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())
        self.menuBar.addAction(self.menuImage.menuAction())
        self.menuBar.addAction(self.menuHelp.menuAction())
        self.fileToolbar.addAction(self.actionNewImage)
        self.fileToolbar.addWidget(self.open_image_button)
        self.fileToolbar.addWidget(self.estimatedKenrel_button)
        self.fileToolbar.addWidget(self.Z_load_button)
        self.fileToolbar.addWidget(self.Z_mask_load_button)
        self.fileToolbar.addAction(self.actionSaveImage)
        self.fileToolbar.addAction(self.actionSaveImageAndData)
        self.fileToolbar.addAction(self.actionDecreaseDisplayZoom)
        self.fileToolbar.addAction(self.actionIncreaseDisplayZoom)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "Explorable SR"+(' (x%d)'%(self.display_zoom_factor) if self.display_zoom_factor>1 else '')))
        MainWindow.setWindowTitle(_translate("MainWindow", "Explorable SR"))
        self.menuFIle.setTitle(_translate("MainWindow", "FIle"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuImage.setTitle(_translate("MainWindow", "Image"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.fileToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        # self.drawingToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        # self.fontToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        # self.actionCopy.setText(_translate("MainWindow", "Copy"))
        # self.actionCopy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        # self.actionClearImage.setText(_translate("MainWindow", "Clear Image"))
        # self.actionOpenImage.setText(_translate("MainWindow", "Open Image..."))
        self.actionSaveImage.setText(_translate("MainWindow", "Save Image As..."))
        self.actionSaveImageAndData.setText(_translate("MainWindow", "Save Image & Z map"))
        self.actionInvertColors.setText(_translate("MainWindow", "Invert Colors"))
        self.actionFlipHorizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        self.actionFlipVertical.setText(_translate("MainWindow", "Flip Vertical"))
        self.actionNewImage.setText(_translate("MainWindow", "New Image"))
        self.actionBold.setText(_translate("MainWindow", "Bold"))
        self.actionBold.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionItalic.setText(_translate("MainWindow", "Italic"))
        self.actionItalic.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionUnderline.setText(_translate("MainWindow", "Underline"))
        self.actionFillShapes.setText(_translate("MainWindow", "Fill Shapes?"))

