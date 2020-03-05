from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets
import numpy as np
import os

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

    def Define_Grid_layout(self, layout_name, buttons_list, height_width_ratio=None, layout_cols=None,parent=None):
        print('Adding toolbar %s'%(layout_name))
        num_buttons = len(buttons_list)
        if layout_cols is None:
            layout_cols = np.maximum(1, int(np.round(np.sqrt(num_buttons / height_width_ratio))))
        layout_rows = int(np.ceil(num_buttons/layout_cols))
        buttons_list += [(None,1,1) for i in range(layout_rows*layout_cols-len(buttons_list))]
        buttons_list = np.reshape(buttons_list, [layout_rows, layout_cols, 3])
        cum_locations = 1*buttons_list[:, :, 1:]
        cum_locations[:,:,0] = np.maximum(cum_locations[:,:,0],np.max(cum_locations[:,:,0],1,keepdims=True))
        cum_locations[:,:,1] = np.maximum(cum_locations[:,:,1],np.max(cum_locations[:,:,1],0,keepdims=True))
        cum_locations = np.stack([np.concatenate([np.zeros([1, layout_cols]).astype(int), np.cumsum(cum_locations[:-1, :, 0], 0)], 0),
                                  np.concatenate([np.zeros([layout_rows, 1]).astype(int), np.cumsum(cum_locations[:, :-1, 1], 1)], 1)], -1)
        widget = QtWidgets.QWidget()
        title = QLabel(parent=widget)
        title.setText(layout_name.replace('_',' '))
        setattr(self, layout_name, QtWidgets.QGridLayout(widget))
        new_layout = getattr(self, layout_name)
        new_layout.setContentsMargins(0, title.height(), 0 ,0)
        new_layout.setSpacing(15)
        new_layout.setObjectName(layout_name)
        for r in range(buttons_list.shape[0]):
            for c in range(buttons_list.shape[1]):
                if (r+1)*(c+1)>num_buttons:
                    break
                new_layout.addWidget(buttons_list[r,c,0],  cum_locations[r,c,0],cum_locations[r,c,1], buttons_list[r,c,1], buttons_list[r,c,2])
        return widget

    def Define_Nesting_Layout(self,parent,horizontal,name):
        if horizontal:
            new_layout = QtWidgets.QHBoxLayout()
        else:
            new_layout = QtWidgets.QVBoxLayout()
        new_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        new_layout.setSpacing(6)
        new_layout.setObjectName(name)
        parent.addLayout(new_layout)
        return new_layout

    # def setupUi(self, MainWindow):
    def setupUi(self):
        # self = MainWindow
        MainWindow = self
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000,1000)

        # Configuring parameters:
        self.max_SVD_Lambda = MAX_SVD_LAMBDA
        self.display_ESRGAN = DISPLAY_ESRGAN_RESULTS
        self.button_size = DEFAULT_BUTTON_SIZE

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setSizePolicy(ReturnSizePolicy(QtWidgets.QSizePolicy.Maximum,self.centralWidget.sizePolicy().hasHeightForWidth()))
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        # self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # self.horizontalLayout = self.Define_Nesting_Layout(self.centralWidget,horizontal=True,name='horizontalLayout')
        # self.verticalLayout_L = QtWidgets.QVBoxLayout()
        # self.verticalLayout_L.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        # self.verticalLayout_L.setSpacing(6)
        # self.verticalLayout_L.setObjectName("verticalLayout_L")
        # self.verticalLayout_R = QtWidgets.QVBoxLayout()
        # self.verticalLayout_R.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        # self.verticalLayout_R.setSpacing(6)
        # self.verticalLayout_R.setObjectName("verticalLayout_R")
        # self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_2.setSpacing(6)
        # self.verticalLayout_2.setObjectName("verticalLayout_2")
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
        self.canvas.sliderZ0.setOrientation(Qt.Horizontal)
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
        self.canvas.sliderZ1.setOrientation(Qt.Horizontal)
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

        def Define_Action_Button(button_name,tooltip,disabled=False,checkable=False,size=1):
            setattr(self,button_name+'_action',QtWidgets.QToolButton(icon=QIcon(QPixmap('icons/'+button_name.replace('_action','_button')+'.png'))))
            button = getattr(self,button_name+'_action')
            button.setObjectName(button_name+'_action')
            button.setEnabled(not disabled)
            button.setToolTip(tooltip)
            button.setCheckable(checkable)
            if not isinstance(size,list):
                size = [size,size]
            button.setMinimumSize(QtCore.QSize(size[0]*self.button_size,size[1]*self.button_size))
            button.setMaximumSize(QtCore.QSize(size[0]*self.button_size,size[1]*self.button_size))

        # if self.USE_LAYOUTS_METHOD:
        #     Z_tool_bar = self.Define_Grid_layout(layout_name='ZToolbar', parent=self.verticalLayout_2,
        #                             buttons_list=[],
        #                             height_width_ratio=1)
        #     self.verticalLayout_2.addWidget(Z_tool_bar)
        #     self.addToolBarBreak()


        Define_Push_Button('uniformZ', tooltip='Spatially uniform Z', disabled=False, checkable=False)
        # self.gridLayout.addWidget(self.uniformZ_button,2,1, 1, 1)

        Define_Push_Button('Zdisplay', tooltip='Toggle Z display', disabled=False, checkable=True)
        # self.gridLayout.addWidget(self.Zdisplay_button, 2,0, 1, 1)


        # Imprinting translation buttons:
        imprint_translations = ['left','right','up','down']
        for button_num,button in enumerate(imprint_translations):
            Define_Push_Button(button+'_imprinting', tooltip='Move imprinting '+button, disabled=True)
            self.gridLayout.addWidget(getattr(self,button+'_imprinting_button'), 4, button_num, 1, 1)
        # Imprinting dimensions change buttons:
        imprint_stretches = ['narrower','wider','taller','shorter']
        for button_num,button in enumerate(imprint_stretches):
            Define_Push_Button(button+'_imprinting', tooltip='Make imprinting '+button, disabled=True)
            self.gridLayout.addWidget(getattr(self,'%s_imprinting_button'%(button)), 5, button_num, 1, 1)

        self.auto_hist_temperature_mode_Enabled = False
        if self.auto_hist_temperature_mode_Enabled:
            Define_Push_Button('auto_hist_temperature_mode', tooltip='Automatic histogram temperature', checkable=True)
            self.gridLayout.addWidget(self.auto_hist_temperature_mode_button, 2, 2, 1, 1)

        # self.verticalLayout_2.addWidget(self.widget)
        # spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.verticalLayout_2.addItem(spacerItem)
        # self.horizontalLayout.addLayout(self.verticalLayout_2)
        # self.phisical_canvas = QtWidgets.QLabel(self.centralWidget)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.phisical_canvas.sizePolicy().hasHeightForWidth())
        # self.phisical_canvas.setSizePolicy(sizePolicy)
        # self.phisical_canvas.setText("")
        # self.phisical_canvas.setObjectName("phisical_canvas")
        # self.horizontalLayout.addWidget(self.phisical_canvas)
        # self.horizontalLayout.addLayout(self.verticalLayout_L)
        # self.horizontalLayout.addLayout(self.verticalLayout_R)
        # self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        # self.horizontalLayout_2.setSpacing(6)
        # self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        # spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout_2.addItem(spacerItem1)
        # self.horizontalLayout.addLayout(self.horizontalLayout_2)
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
        # self.fileToolbar = QtWidgets.QToolBar(MainWindow)
        # self.fileToolbar.setIconSize(QtCore.QSize(16, 16))
        # self.fileToolbar.setObjectName("fileToolbar")
        # MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.fileToolbar)

        # MainWindow.addToolBarBreak()#.addItem(spacerItem2)

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

        Define_Action_Button(button_name='scribble_reset',tooltip='Erase scribble in region')
        Define_Action_Button(button_name='apply_scribble',tooltip='Perform a single scribble/imprinting application step',disabled=True)
        Define_Action_Button(button_name='loop_apply_scribble',tooltip='Perform multiple scribble/imprinting application steps',disabled=True)
        Define_Push_Button(button_name='im_input',tooltip='Set imprinting rectangle (Using transparent color)',checkable=True,disabled=True)
        Define_Push_Button(button_name='im_input_auto_location',tooltip='Set boundaries for automatic imprinting location (Using transparent color)',checkable=True,disabled=True)
        Define_Action_Button(button_name='open_image',tooltip='Load LR image')
        Define_Action_Button(button_name='Z_load',tooltip='Load Z map')
        Define_Push_Button('estimatedKenrel',tooltip='Use estimated SR kernel',checkable=True)
        Define_Action_Button('ProcessRandZ',tooltip='Produce random images')
        Define_Action_Button('ProcessLimitedRandZ',tooltip='Produce random images close to current')
        Define_Push_Button('special_behavior',tooltip='Toggle special behavior',checkable=True)
        Define_Action_Button('IncreaseSTD',tooltip='Increase local STD (Magnitude)')
        Define_Action_Button('DecreaseSTD',tooltip='Decrease local STD (Magnitude)')

        self.STD_increment = QtWidgets.QDoubleSpinBox(MainWindow)
        self.STD_increment.setObjectName("STD_increment")
        self.STD_increment.setValue(0.03)
        self.STD_increment.setRange(0,0.3)
        self.STD_increment.setSingleStep(0.01)
        self.STD_increment.setDecimals(2)
        self.STD_increment.setToolTip('STD/Brightness change')

        Define_Action_Button('DecreaseTV',tooltip='Decrease TV')
        Define_Action_Button('ImitateHist',tooltip='Encourage desired histogram',disabled=True)
        Define_Action_Button('ImitatePatchHist',tooltip='Encourage desired mean-less (normalized) patches',disabled=True)
        Define_Action_Button('FoolAdversary',tooltip='Fool discriminator')
        Define_Action_Button('IncreasePeriodicity_2D',tooltip='Increase 2D periodicity',disabled=True)
        Define_Action_Button('IncreasePeriodicity_1D',tooltip='Increase 1D periodicity',disabled=True)
        Define_Push_Button(button_name='indicatePeriodicity', tooltip='Set desired periodicity', checkable=True)
        Define_Action_Button('SaveImageAndData',tooltip='Save image, Z and scribble data')
        Define_Push_Button('DecreaseDisplayZoom',tooltip='Decrease zoom')
        Define_Push_Button('IncreaseDisplayZoom',tooltip='Increase zoom')
        Define_Action_Button('SaveImage',tooltip='Save image')
        Define_Action_Button('open_HR_image',tooltip='Synthetically downscale an HR image')
        Define_Action_Button('selectrect', tooltip='Rectangle selection', checkable=True)
        Define_Action_Button('selectpoly', tooltip='Polygon selection', checkable=True)
        Define_Action_Button('unselect', tooltip='De-select', disabled=False, checkable=False)
        Define_Action_Button('invertSelection', tooltip='Invert selection', disabled=False, checkable=False)
        Define_Action_Button(button_name='Z_mask_load',tooltip='Load Z map to infer selection')
        Define_Push_Button('desiredExternalAppearanceMode', tooltip='Select external desired region', disabled=False, checkable=True)
        Define_Push_Button('desiredAppearanceMode', tooltip='Select desired region within image', disabled=False, checkable=True)
        Define_Action_Button('undoZ', tooltip='Undo image manipulation', disabled=True)
        Define_Action_Button('redoZ', tooltip='Redo image manipulation', disabled=True)
        Define_Action_Button(button_name='CopyFromAlternative',tooltip='Copy displayed random region to Z',disabled=True)
        Define_Action_Button(button_name='Copy2Alternative', tooltip='Copy region from Z to random images')
        Define_Push_Button(button_name='pencil',tooltip='Pencil',checkable=True)
        Define_Push_Button(button_name='dropper',tooltip='Eyedropper',checkable=True)
        Define_Push_Button(button_name='line',tooltip='Straight line drawing',checkable=True)
        Define_Push_Button(button_name='polygon',tooltip='Polygon drawing',checkable=True)
        Define_Push_Button(button_name='rect',tooltip='Rectangle drawing',checkable=True)
        Define_Push_Button(button_name='ellipse',tooltip='Ellipse drawing',checkable=True)
        self.canvas.primary_button = QtWidgets.QPushButton(self.Scribble_Toolbar)
        self.canvas.primary_button.setMinimumSize(QtCore.QSize(40, 40))
        self.canvas.primary_button.setMaximumSize(QtCore.QSize(40, 40))
        self.canvas.primary_button.setText("")
        self.canvas.primary_button.setObjectName("primary_button")
        color_state_cycle_icon = QIcon()
        color_state_cycle_icon.addPixmap(QPixmap("icons/color_state_cycle.png"), QIcon.Normal, QIcon.Off)
        self.canvas.cycleColorState_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=color_state_cycle_icon)
        self.canvas.cycleColorState_button.setCheckable(False)
        sizeicon = QLabel()
        sizeicon.setPixmap(QPixmap(os.path.join('icons', 'border-weight.png')))
        self.sizeselect = QSlider()
        self.sizeselect.setRange(1, 20)
        self.sizeselect.setOrientation(Qt.Horizontal)
        self.sizeselect.valueChanged.connect(lambda s: self.canvas.set_config('size', s))
        Define_Action_Button('undo_scribble', tooltip='Undo scribble/imprint', disabled=True)

        Define_Action_Button('redo_scribble', tooltip='Redo scribble/imprint', disabled=True)

        if self.USE_LAYOUTS_METHOD:
            self.fileToolbar = None
            # self.temp_layout = QtWidgets.QHBoxLayout()
            load_and_save = self.Define_Grid_layout(layout_name='Load & Save', parent=self.fileToolbar,
                                    buttons_list=[(self.open_image_action,1,1),(self.open_HR_image_action,1,1),(self.Z_load_action,1,1),(self.SaveImageAndData_action,1,1)],
                                    height_width_ratio=0.25)
            display_TB = self.Define_Grid_layout(layout_name='Display',parent=self.fileToolbar,
                                    buttons_list=[(self.Zdisplay_button,1,1),(self.IncreaseDisplayZoom_button,1,1),(self.DecreaseDisplayZoom_button,1,1),(self.DisplayedImageSelection_button,1,4),
                                                  (self.CopyFromAlternative_action, 1, 1),(self.Copy2Alternative_action, 1, 1)],
                                    layout_cols=4)
            # self.addToolBarBreak()
            uniform_Z_control_TB = self.Define_Grid_layout(layout_name='Uniform Z control',buttons_list=[(self.canvas.sliderZ0,1,4), (self.canvas.sliderZ1,1,4),
                                                  (self.canvas.slider_third_channel,4,4),(self.uniformZ_button,1,1)],layout_cols=2)
            region_selection_TB = self.Define_Grid_layout('Region selection',
                buttons_list=[(self.selectrect_action, 1, 1),(self.invertSelection_action, 1, 1),(self.Z_mask_load_action, 1, 1),(self.selectpoly_action, 1, 1),(self.unselect_action, 1, 1)],layout_cols=3)
            reference_region_TB = self.Define_Grid_layout('Reference region',buttons_list=[(self.desiredAppearanceMode_button,1,1),(self.desiredExternalAppearanceMode_button,1,1)],layout_cols=1)
            periodicity_TB = self.Define_Grid_layout('Periodicity',buttons_list=[(self.periodicity_mag_1,1,4), (self.periodicity_mag_2,1,4),(self.indicatePeriodicity_button, 1, 1),
                                                                              (self.IncreasePeriodicity_1D_action,1,1),(self.IncreasePeriodicity_2D_action,1,1)],layout_cols=2)
            general_TB = self.Define_Grid_layout('General',
                buttons_list=[(self.estimatedKenrel_button, 1, 1),(self.special_behavior_button, 1, 1),(self.undoZ_action, 1, 1),(self.redoZ_action, 1, 1)],layout_cols=2)
            optimize_Z_TB = self.Define_Grid_layout('Optimize Z',buttons_list=[(self.IncreaseSTD_action,1,1),(self.DecreaseSTD_action,1,1),(self.DecreaseTV_action,1,1),
                (self.ImitateHist_action,1,1),(self.ImitatePatchHist_action,1,1),(self.FoolAdversary_action,1,1),(self.STD_increment,1,1),(self.ProcessRandZ_action,1,1),
                                                                               (self.ProcessLimitedRandZ_action,1,1)],layout_cols=4)
            scribble_A_TB = self.Define_Grid_layout('Scribbling',buttons_list=[(getattr(self,m+'_button'),1,1) for m in ['pencil','line', 'polygon','ellipse', 'rect']],layout_cols=1)
            scribble_B_TB = self.Define_Grid_layout('Scribble B',buttons_list=[(self.dropper_button,1,1),(self.canvas.primary_button,1,1),(self.canvas.cycleColorState_button,1,1),
                (sizeicon,1,1),(self.sizeselect,1,1),(self.scribble_reset_action,1,1),(self.apply_scribble_action,1,1),(self.loop_apply_scribble_action,1,1),
                                           (self.undo_scribble_action,1,1),(self.redo_scribble_action,1,1)],layout_cols=1)
            imprinting_TB = self.Define_Grid_layout('Imprinting',buttons_list=[(self.im_input_button,1,1),(self.im_input_auto_location_button,1,1)]+\
                [(getattr(self,button+'_imprinting_button'),1,1) for button in (imprint_stretches+imprint_translations)],layout_cols=4)
            temporary = self.Define_Grid_layout(layout_name='Temporary', parent=self.fileToolbar,
                                    buttons_list=[(self.SaveImage_action, 1, 1),],height_width_ratio=1 / 3)

            self.verticalLayout_L = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_L')
            self.verticalLayout_C = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_C')
            self.verticalLayout_R = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_R')
            self.verticalLayout_L.addWidget(load_and_save)
            self.verticalLayout_L.addWidget(display_TB)
            self.verticalLayout_L.addWidget(temporary)
            self.verticalLayout_L.addWidget(region_selection_TB)
            self.verticalLayout_L.addWidget(general_TB)
            self.verticalLayout_L.addWidget(reference_region_TB)
            self.verticalLayout_L.addWidget(uniform_Z_control_TB)
            self.verticalLayout_C.addWidget(scribble_A_TB)
            self.verticalLayout_C.addWidget(scribble_B_TB)
            self.verticalLayout_R.addWidget(periodicity_TB)
            self.verticalLayout_R.addWidget(optimize_Z_TB)
            self.verticalLayout_R.addWidget(imprinting_TB)
            # self.verticalLayout_L.addLayout(self.temp_layout)
        else:
            self.gridLayout.addWidget(self.redoZ_action, 3, 1, 1, 1)
            self.gridLayout.addWidget(self.undoZ_action, 3, 0, 1, 1)
            self.gridLayout.addWidget(self.selectpoly_action, 0, 0, 1, 1)
            self.gridLayout.addWidget(self.selectrect_action, 0, 1, 1, 1)
            self.gridLayout.addWidget(self.unselect_action, 0, 2, 1, 1)
            self.gridLayout.addWidget(self.invertSelection_action, 0, 3, 1, 1)
            self.gridLayout.addWidget(self.desiredExternalAppearanceMode_button, 1, 0, 1, 1)
            self.gridLayout.addWidget(self.desiredAppearanceMode_button, 1, 1, 1, 1)
            self.fileToolbar.addWidget(self.open_HR_image_action)
            self.fileToolbar.addWidget(self.open_image_action)
            self.fileToolbar.addWidget(self.Z_load_action)
            self.fileToolbar.addWidget(self.SaveImageAndData_action)
            self.fileToolbar.addWidget(self.DecreaseDisplayZoom_button)
            self.fileToolbar.addWidget(self.IncreaseDisplayZoom_button)
            self.gridLayout.addWidget(self.Zdisplay_button, 2,0, 1, 1)
            self.gridLayout.addWidget(self.uniformZ_button, 2, 1, 1, 1)

            self.fileToolbar.addWidget(self.estimatedKenrel_button)
            self.fileToolbar.addWidget(self.Z_mask_load_action)
            self.fileToolbar.addWidget(self.SaveImage_action)
            # Scribble:
            self.Scribble_Toolbar.addWidget(self.scribble_reset_action)
            self.Scribble_Toolbar.addWidget(sizeicon)
            self.Scribble_Toolbar.addWidget(self.sizeselect)

            self.Scribble_Toolbar.addWidget(self.im_input_button)
            self.Scribble_Toolbar.addWidget(self.im_input_auto_location_button)
            self.Scribble_Toolbar.addWidget(self.dropper_button)
            self.Scribble_Toolbar.addWidget(self.pencil_button)
            self.Scribble_Toolbar.addWidget(self.line_button)
            self.Scribble_Toolbar.addWidget(self.ellipse_button)
            self.Scribble_Toolbar.addWidget(self.polygon_button)
            self.Scribble_Toolbar.addWidget(self.rect_button)
            self.Scribble_Toolbar.addWidget(self.apply_scribble_action)
            self.Scribble_Toolbar.addWidget(self.loop_apply_scribble_action)
            self.Scribble_Toolbar.addWidget(self.canvas.cycleColorState_button)
            self.Scribble_Toolbar.addWidget(self.canvas.primary_button)
            self.gridLayout.addWidget(self.undo_scribble_action, 3, 2, 1, 1)
            self.gridLayout.addWidget(self.redo_scribble_action, 3, 3, 1, 1)

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
        # self.fileToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))

        # self.drawingToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        # self.fontToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        # self.actionCopy.setText(_translate("MainWindow", "Copy"))
        # self.actionCopy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        # self.actionClearImage.setText(_translate("MainWindow", "Clear Image"))
        # self.actionOpenImage.setText(_translate("MainWindow", "Open Image..."))
        # self.actionSaveImage.setText(_translate("MainWindow", "Save Image As..."))
        # self.actionSaveImageAndData.setText(_translate("MainWindow", "Save Image & Z map"))
        # self.actionInvertColors.setText(_translate("MainWindow", "Invert Colors"))
        # self.actionFlipHorizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        # self.actionFlipVertical.setText(_translate("MainWindow", "Flip Vertical"))
        # self.actionNewImage.setText(_translate("MainWindow", "New Image"))
        # self.actionBold.setText(_translate("MainWindow", "Bold"))
        # self.actionBold.setShortcut(_translate("MainWindow", "Ctrl+B"))
        # self.actionItalic.setText(_translate("MainWindow", "Italic"))
        # self.actionItalic.setShortcut(_translate("MainWindow", "Ctrl+I"))
        # self.actionUnderline.setText(_translate("MainWindow", "Underline"))
        # self.actionFillShapes.setText(_translate("MainWindow", "Fill Shapes?"))

