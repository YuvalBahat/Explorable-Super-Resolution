from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets
import numpy as np
import os
from scipy.signal import convolve2d

MAX_SVD_LAMBDA = 1.
DEFAULT_BUTTON_SIZE = 30
ENABLE_ADVERSARY_BUTTON_IN_SR = False

def ReturnSizePolicy(policy,hasHeightForWidth):
    sizePolicy = QtWidgets.QSizePolicy(policy, policy)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(hasHeightForWidth)
    return sizePolicy


class Ui_MainWindow(object):
    def Greedily_Find_Location(self, occupancy_map, button_size):
        occupancy_map = np.pad(np.logical_not(occupancy_map),((0,button_size[0]-1),(0,button_size[1]-1)))
        optional_locations = np.abs(convolve2d(occupancy_map,np.ones(button_size)/np.prod(button_size),mode='valid')-1)<1e-5
        return np.unravel_index(np.argwhere(optional_locations.reshape([-1]))[0][0],optional_locations.shape)

    def Define_Grid_layout(self, layout_name, buttons_list, height_width_ratio=None, layout_cols=None):
        print('Adding toolbar %s'%(layout_name))
        num_buttons = len(buttons_list)
        button_sizes = np.stack([np.array([b.height(),b.width()])//self.button_size if b is not None else np.array([1,1]) for b in buttons_list])
        assert np.all(button_sizes.reshape([-1])>0),'Button size must be at least 1'
        if layout_cols is None:
            layout_cols = np.maximum(1, int(np.round(np.sqrt(num_buttons / height_width_ratio))))
        assert max([s[1] for s in button_sizes])<=layout_cols,'A button is wider than the number of layout columns'
        widget = QtWidgets.QWidget()
        title = QLabel(parent=widget)
        title.setText(layout_name.replace('_',' '))
        title.setMinimumSize(title.sizeHint())
        setattr(self, layout_name, QtWidgets.QGridLayout(widget))
        new_layout = getattr(self, layout_name)
        new_layout.setContentsMargins(0, title.height(), 0 ,0)
        new_layout.setSpacing(5)
        new_layout.setObjectName(layout_name)
        max_button_height = max([s[0] for s in button_sizes])
        occupancy_map = np.zeros([max_button_height,layout_cols]).astype(np.bool)
        for button_num in range(num_buttons):
            cur_row,cur_col = self.Greedily_Find_Location(occupancy_map=occupancy_map, button_size=button_sizes[button_num])
            occupancy_map[cur_row:cur_row+button_sizes[button_num,0],cur_col:cur_col+button_sizes[button_num,1]] = True
            lowest_partially_occupied_row = np.argwhere(np.any(occupancy_map,1))[-1][0]
            occupancy_map = np.concatenate([occupancy_map[:lowest_partially_occupied_row+1,:],np.zeros([max_button_height,layout_cols]).astype(np.bool)],0)
            if buttons_list[button_num] is not None:
                new_layout.addWidget(buttons_list[button_num], cur_row, cur_col, button_sizes[button_num, 0],button_sizes[button_num, 1])
        return widget

    def Define_Nesting_Layout(self,parent,horizontal,name,title=None):
        if horizontal:
            new_layout = QtWidgets.QHBoxLayout()
        else:
            new_layout = QtWidgets.QVBoxLayout()
        new_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        new_layout.setSpacing(10)
        new_layout.setObjectName(name)
        parent.addLayout(new_layout)
        if title is not None:
            title_widget = QWidget()
            title_label = QLabel(parent=title_widget)
            title_label.setText(title)
            new_layout.addWidget(title_widget)
        return new_layout

    def Set_Button_Size(self,button,size):
        Qsize_obj = QtCore.QSize(size[0] * self.button_size, size[1] * self.button_size)
        button.setMinimumSize(Qsize_obj)
        button.setMaximumSize(Qsize_obj)
        if isinstance(button,QPushButton) or isinstance(button,QToolButton):
            button.setIconSize(0.8*Qsize_obj)

    def Define_Slider(self,slider_name,tooltip,range,position=None,size=1,horizontal=True,parent=None,dial=False):
        if parent is None:
            parent = self
        if dial:
            setattr(parent,slider_name+'_slider',QDial())
        else:
            setattr(parent,slider_name+'_slider',QSlider())
        slider = getattr(parent,slider_name+'_slider')
        if dial:
            slider.setWrapping(True)
            slider.setNotchesVisible(True)
        slider.setObjectName(slider_name+'_slider')
        slider.setToolTip(tooltip)
        slider.setRange(range[0],range[1])
        if position is not None:
            slider.setSliderPosition(position)
        slider.setOrientation(Qt.Horizontal if horizontal else Qt.Vertical)
        if not isinstance(size,list):
            if dial:
                size = [size,size]
            else:
                size = [size,1] if horizontal else [1,size]
        self.Set_Button_Size(slider,size)

    def Define_Button(self,button_name,tooltip,action_not_push,disabled=False,checkable=False,size=1,parent=None):
        if parent is None:
            parent = self
        if action_not_push:
            setattr(parent, button_name + '_button', QtWidgets.QToolButton(icon=QIcon(QPixmap('icons/' + button_name + '.png'))))
        else:
            setattr(parent, button_name + '_button',QPushButton(icon=QIcon(QPixmap('icons/' + button_name + '.png'))))
        button = getattr(parent,button_name+'_button')
        button.setObjectName(button_name+'_button')
        button.setEnabled(not disabled)
        button.setToolTip(tooltip)
        button.setCheckable(checkable)
        if not isinstance(size,list):
            size = [size,size]
        self.Set_Button_Size(button,size)

    def setupUi(self):
        MainWindow = self
        MainWindow.setObjectName("MainWindow")

        # Configuring parameters:
        self.max_SVD_Lambda = MAX_SVD_LAMBDA
        self.button_size = DEFAULT_BUTTON_SIZE

        # Set parent layouts:
        self.centralWidget = QtWidgets.QWidget()
        self.centralWidget.setSizePolicy(ReturnSizePolicy(QtWidgets.QSizePolicy.Maximum,self.centralWidget.sizePolicy().hasHeightForWidth()))
        self.centralWidget.setObjectName("centralWidget")
        MainWindow.setWindowTitle("Explorable Super Resolution")
        MainWindow.setCentralWidget(self.centralWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # Status bar:
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        ##### Defining buttons:
        self.Define_Button('uniformZ', tooltip='Spatially uniform Z',action_not_push=False, disabled=False, checkable=False)
        self.Define_Button('Zdisplay', tooltip='Toggle Z display',action_not_push=False, disabled=False, checkable=True)
        self.Define_Button(button_name='scribble_reset',tooltip='Erase scribble in region',action_not_push=True)
        self.Define_Button(button_name='apply_scribble',tooltip='Perform a single scribble/imprinting application step',action_not_push=True,disabled=True)
        self.Define_Button(button_name='loop_apply_scribble',tooltip='Perform multiple scribble/imprinting application steps',action_not_push=True,disabled=True)
        self.Define_Button(button_name='imprinting',tooltip='Set imprinting rectangle (Using transparent color)',action_not_push=False,checkable=False,disabled=True)
        self.Define_Button(button_name='imprinting_auto_location',tooltip='Set boundaries for automatic imprinting location (Using transparent color)',action_not_push=False,checkable=False,disabled=True)
        self.Define_Button(button_name='open_image',tooltip='Load LR image',action_not_push=True)
        self.Define_Button(button_name='Z_load',tooltip='Load Z map',action_not_push=True)
        self.Define_Button('estimatedKenrel',tooltip='Use estimated SR kernel',action_not_push=False,checkable=True)
        self.Define_Button('ProcessRandZ',tooltip='Produce random images',action_not_push=True)
        self.Define_Button('ProcessLimitedRandZ',tooltip='Produce random images close to current',action_not_push=True)
        self.Define_Button('special_behavior',tooltip='Toggle special behavior',action_not_push=False,checkable=True)
        self.Define_Button('IncreaseSTD',tooltip='Increase local STD (Magnitude)',action_not_push=True)
        self.Define_Button('DecreaseSTD',tooltip='Decrease local STD (Magnitude)',action_not_push=True)
        self.Define_Button('DecreaseTV',tooltip='Decrease TV',action_not_push=True)
        self.Define_Button('ImitateHist',tooltip='Encourage desired histogram',action_not_push=True,disabled=True)
        self.Define_Button('ImitatePatchHist',tooltip='Encourage desired mean-less (normalized) patches',action_not_push=True,disabled=True)
        self.Define_Button('FoolAdversary',tooltip='Fool discriminator',action_not_push=True)
        self.Define_Button('IncreasePeriodicity_2D',tooltip='Increase 2D periodicity',action_not_push=True,disabled=True)
        self.Define_Button('IncreasePeriodicity_1D',tooltip='Increase 1D periodicity',action_not_push=True,disabled=True)
        self.Define_Button(button_name='indicatePeriodicity', tooltip='Set desired periodicity',action_not_push=False, checkable=True)
        self.Define_Button('SaveImageAndData',tooltip='Save image, Z and scribble data',action_not_push=True)
        self.Define_Button('DecreaseDisplayZoom',tooltip='Decrease zoom',action_not_push=False)
        self.Define_Button('IncreaseDisplayZoom',tooltip='Increase zoom',action_not_push=False)
        # self.Define_Button('SaveImage',tooltip='Save image',action_not_push=True)
        self.Define_Button('open_HR_image',tooltip='Synthetically downscale an HR image',action_not_push=True)
        self.Define_Button('selectrect', tooltip='Rectangle selection',action_not_push=True, checkable=True)
        self.Define_Button('selectpoly', tooltip='Polygon selection',action_not_push=True, checkable=True)
        self.Define_Button('unselect', tooltip='De-select',action_not_push=True, disabled=False, checkable=False)
        self.Define_Button('invertSelection', tooltip='Invert selection',action_not_push=True, disabled=False, checkable=False)
        self.Define_Button(button_name='Z_mask_load',tooltip='Load Z map to infer selection',action_not_push=True)
        self.Define_Button('desiredExternalAppearanceMode', tooltip='Select external desired region',action_not_push=False, disabled=False, checkable=True)
        self.Define_Button('desiredAppearanceMode', tooltip='Select desired region within image',action_not_push=False, disabled=False, checkable=True)
        self.Define_Button('undoZ', tooltip='Undo image manipulation',action_not_push=True, disabled=True)
        self.Define_Button('redoZ', tooltip='Redo image manipulation',action_not_push=True, disabled=True)
        self.Define_Button(button_name='CopyFromAlternative',tooltip='Copy displayed alternative region to Z',action_not_push=True,disabled=True)
        self.Define_Button(button_name='Copy2Alternative', tooltip='Copy region from Z to alternative images',action_not_push=True)
        self.Define_Button(button_name='pencil',tooltip='Pencil',action_not_push=False,checkable=True)
        self.Define_Button(button_name='dropper',tooltip='Eyedropper',action_not_push=False,checkable=True)
        self.Define_Button(button_name='line',tooltip='Straight line drawing',action_not_push=False,checkable=True)
        self.Define_Button(button_name='polygon',tooltip='Polygon drawing',action_not_push=False,checkable=True)
        self.Define_Button(button_name='rect',tooltip='Rectangle drawing',action_not_push=False,checkable=True)
        self.Define_Button(button_name='ellipse',tooltip='Ellipse drawing',action_not_push=False,checkable=True)
        self.Define_Button('color',tooltip='Scribble color',action_not_push=False,size=1,parent=self.canvas)
        self.Define_Button(button_name='cycleColorState',tooltip='Cycle through scribble modes',action_not_push=False,parent=self.canvas)
        self.Define_Button('undo_scribble', tooltip='Undo scribble/imprint',action_not_push=True, disabled=True)
        self.Define_Button('redo_scribble', tooltip='Redo scribble/imprint',action_not_push=True, disabled=True)

        # Imprinting translation buttons:
        imprint_translations = ['left','right','up','down']
        for button_num,button in enumerate(imprint_translations):
            self.Define_Button(button+'_imprinting', tooltip='Move imprinting '+button,action_not_push=False, disabled=True)

        # Imprinting dimensions change buttons:
        imprint_stretches = ['narrower','wider','taller','shorter']
        for button_num,button in enumerate(imprint_stretches):
            self.Define_Button(button+'_imprinting', tooltip='Make imprinting '+button,action_not_push=False, disabled=True)

        imprint_rotations = ['clockwise','counter_clockwise']
        for button_num,button in enumerate(imprint_rotations):
            self.Define_Button(button+'_imprinting', tooltip='Make imprinting '+button,action_not_push=False, disabled=True)

        self.auto_hist_temperature_mode_Enabled = False
        if self.auto_hist_temperature_mode_Enabled:
            self.Define_Button('auto_hist_temperature_mode', tooltip='Automatic histogram temperature',action_not_push=False, checkable=True)

        # Define uniform Z control and other sliders:
        self.Define_Slider('Z0',tooltip='Primary direction gradients magnitude',range=[0, 100*self.max_SVD_Lambda],position=100*self.max_SVD_Lambda/2,size=3,parent=self.canvas,horizontal=False)
        self.Define_Slider('Z1',tooltip='Secondary direction gradients magnitude',range=[0, 100*self.max_SVD_Lambda],position=100*self.max_SVD_Lambda/2,size=3,parent=self.canvas,horizontal=False)
        self.Define_Slider('third_channel',tooltip='Graidents direction',range=[-100*np.pi, 100*np.pi],position=0,size=3,parent=self.canvas,dial=True)
        self.Define_Slider('sizeselect',tooltip='Set line width',range=[1,20],size=2,horizontal=True)

        if self.JPEG_GUI:
            self.Define_Button('H_clockwise',tooltip='Hue clockwise',action_not_push=True)
            self.Define_Button('H_counter_clockwise', tooltip='Hue counter-clockwise', action_not_push=True)
            self.Define_Button('S_up', tooltip='Increase saturation', action_not_push=True)
            self.Define_Button('S_down', tooltip='Decrease saturation', action_not_push=True)
            self.Define_Button('V_up', tooltip='Increase value', action_not_push=True)
            self.Define_Button('V_down', tooltip='Decrease value', action_not_push=True)

        # Define button controlling displayed image version:
        self.DisplayedImageSelection_button = QtWidgets.QComboBox()
        self.DisplayedImageSelection_button.setObjectName("DisplayedImageSelection_button")
        self.DisplayedImageSelection_button.setEnabled(False)
        self.DisplayedImageSelection_button.setToolTip('Displayed image')
        self.Set_Button_Size(self.DisplayedImageSelection_button,[3,1])
        self.DisplayedImageSelection_button.highlighted.connect(self.SelectImage2Display)
        if self.display_ESRGAN:
            self.DisplayedImageSelection_button.addItem('ESRGAN')
            self.DisplayedImageSelection_button.setEnabled(True)
            self.ESRGAN_index = self.DisplayedImageSelection_button.findText('ESRGAN')
        else:
            self.ESRGAN_index = None
        # I always add GT display, and only enable it for images with GT
        self.DisplayedImageSelection_button.addItem('GT')
        self.DisplayedImageSelection_button.setEnabled(True)
        self.GT_HR_index = self.DisplayedImageSelection_button.findText('GT')
        self.DisplayedImageSelection_button.addItem('Z')
        self.cur_Z_im_index = self.DisplayedImageSelection_button.findText('Z')
        self.canvas.cur_Z_im_index = self.cur_Z_im_index
        self.DisplayedImageSelection_button.addItems([str(i+1) for i in range(self.num_random_Zs)])
        self.random_display_indexes = [self.DisplayedImageSelection_button.findText(str(i+1)) for i in range(self.num_random_Zs)]
        self.DisplayedImageSelection_button.addItem('Scribble')
        self.canvas.scribble_display_index = self.DisplayedImageSelection_button.findText('Scribble')
        self.DisplayedImageSelection_button.addItem('Input')
        self.input_display_index = self.DisplayedImageSelection_button.findText('Input')

        ######## Defining user input boxes:
        if self.JPEG_GUI:
        #     Quality-Factor selector
            self.QF_box = QtWidgets.QDoubleSpinBox()
            self.QF_box.setObjectName("QF_box")
            self.QF_box.setValue(10)
            # self.QF_box.setMaximum(MAXIMAL_JPEG_QF)
            self.QF_box.setMinimum(5)
            self.QF_box.setSingleStep(1)
            self.Set_Button_Size(self.QF_box, [2, 1])
            self.QF_box.setDecimals(0)
            self.QF_box.setToolTip('JPEG Quality Factor')

        # Weight limiting random Z generated images, if enabled:
        self.randomLimitingWeightBox_Enabled = False
        if self.randomLimitingWeightBox_Enabled:
            self.randomLimitingWeightBox = QtWidgets.QDoubleSpinBox()
            self.randomLimitingWeightBox.setObjectName("randomLimitingWeightBox")
            self.randomLimitingWeightBox.setValue(1.)
            self.randomLimitingWeightBox.setMaximum(200)
            self.randomLimitingWeightBox.setToolTip('Random images limitation weight')

        # Periodicity related input boxes:
        self.periodicity_mag_1_button = QtWidgets.QDoubleSpinBox()
        self.periodicity_mag_1_button.setObjectName("periodicity_mag_1_button")
        self.periodicity_mag_1_button.setValue(1.)
        self.periodicity_mag_1_button.setMaximum(200)
        self.periodicity_mag_1_button.setSingleStep(0.1)
        self.periodicity_mag_1_button.setDecimals(1)
        self.periodicity_mag_1_button.setToolTip('Primary period length')
        self.Set_Button_Size(self.periodicity_mag_1_button,[2,1])
        self.periodicity_mag_2_button = QtWidgets.QDoubleSpinBox()
        self.periodicity_mag_2_button.setObjectName("periodicity_mag_2_button")
        self.periodicity_mag_2_button.setValue(1.)
        self.periodicity_mag_2_button.setMaximum(200)
        self.periodicity_mag_2_button.setSingleStep(0.1)
        self.periodicity_mag_2_button.setDecimals(1)
        self.periodicity_mag_2_button.setToolTip('Secondary period length')
        self.Set_Button_Size(self.periodicity_mag_2_button,[2,1])

        # STD modification degree input box:
        self.STD_increment = QtWidgets.QDoubleSpinBox()
        self.STD_increment.setObjectName("STD_increment")
        self.STD_increment.setValue(0.03)
        self.STD_increment.setRange(0,0.3)
        self.STD_increment.setSingleStep(0.01)
        self.STD_increment.setDecimals(2)
        self.STD_increment.setToolTip('STD/Brightness change')
        self.Set_Button_Size(self.STD_increment,[2,1])


        ###### Defining layouts holding groups of buttons:
        load_and_save = self.Define_Grid_layout(layout_name='Load & Save',
            buttons_list=[self.open_image_button,self.open_HR_image_button,self.Z_load_button,self.SaveImageAndData_button]+([self.QF_box] if self.JPEG_GUI else []),
            layout_cols=4)
        display_TB = self.Define_Grid_layout(layout_name='Display',buttons_list=[self.Zdisplay_button,self.IncreaseDisplayZoom_button,self.DecreaseDisplayZoom_button,
            self.DisplayedImageSelection_button,self.CopyFromAlternative_button,self.Copy2Alternative_button],layout_cols=4)
        uniform_Z_control_TB = self.Define_Grid_layout(layout_name='Uniform Z control',buttons_list=[self.canvas.Z0_slider,self.canvas.third_channel_slider, self.canvas.Z1_slider,
                                              self.uniformZ_button],layout_cols=6)
        if self.JPEG_GUI:
            # HSV_TB = self.Define_Grid_layout('HSV',[self.canvas.H_slider,self.canvas.S_slider,self.canvas.V_slider],layout_cols=5)
            HSV_TB = self.Define_Grid_layout('HSV',[self.H_clockwise_button,self.S_up_button,self.V_up_button,
                self.H_counter_clockwise_button,self.S_down_button,self.V_down_button],layout_cols=3)
        region_selection_TB = self.Define_Grid_layout('Region selection',
            buttons_list=[self.selectrect_button,self.invertSelection_button,self.Z_mask_load_button,self.selectpoly_button,self.unselect_button],layout_cols=3)
        periodicity_TB = self.Define_Grid_layout('Periodicity',buttons_list=[self.IncreasePeriodicity_1D_button,self.periodicity_mag_1_button,
              self.IncreasePeriodicity_2D_button, self.periodicity_mag_2_button,self.indicatePeriodicity_button],layout_cols=6)
        general_TB = self.Define_Grid_layout('Reference & General',
            buttons_list=[self.desiredAppearanceMode_button,self.undoZ_button,self.special_behavior_button,self.desiredExternalAppearanceMode_button,
                          self.redoZ_button]+([self.estimatedKenrel_button] if not self.JPEG_GUI else []),layout_cols=3)
        optimize_Z_TB = self.Define_Grid_layout('Optimize Z',buttons_list=[self.IncreaseSTD_button,self.DecreaseSTD_button,self.DecreaseTV_button,
            self.ImitateHist_button,self.ImitatePatchHist_button]+([self.FoolAdversary_button] if ENABLE_ADVERSARY_BUTTON_IN_SR else [])+
            [self.STD_increment,self.ProcessRandZ_button,self.ProcessLimitedRandZ_button]+
            ([self.randomLimitingWeightBox] if self.randomLimitingWeightBox_Enabled else []),layout_cols=4)
        scribbling_tool_buttons = [getattr(self,m+'_button') for m in ['pencil','line', 'polygon','ellipse', 'rect']]
        sizeicon = QLabel(size=QSize(self.button_size,self.button_size))
        sizeicon.setPixmap(QPixmap(os.path.join('icons', 'border-weight.png')))
        scribble_A_TB = self.Define_Grid_layout('Scribbling',buttons_list=[sizeicon,self.sizeselect_slider]+scribbling_tool_buttons,layout_cols=4)
        scribble_B_TB = self.Define_Grid_layout('Manage scribble',buttons_list=[self.dropper_button,self.scribble_reset_button,self.undo_scribble_button,self.canvas.color_button,
            self.apply_scribble_button,self.redo_scribble_button,self.canvas.cycleColorState_button,self.loop_apply_scribble_button],layout_cols=3)
        imprinting_TB = self.Define_Grid_layout('Imprinting',buttons_list=[self.imprinting_button,self.imprinting_auto_location_button,None,None]+\
            [getattr(self,button+'_imprinting_button') for button in (imprint_stretches+imprint_translations+imprint_rotations)],layout_cols=4)


        #### Assemble layouts together to form GUI:
        self.verticalLayout_L = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_L')
        self.verticalLayout_C = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_C')
        self.verticalLayout_R = self.Define_Nesting_Layout(self.horizontalLayout, horizontal=False,name='verticalLayout_R')
        self.verticalLayout_L.addWidget(load_and_save)
        self.verticalLayout_L.addWidget(display_TB)
        self.verticalLayout_L.addWidget(region_selection_TB)
        self.verticalLayout_L.addWidget(general_TB)
        self.verticalLayout_C.addWidget(scribble_A_TB)
        self.verticalLayout_C.addWidget(scribble_B_TB)
        self.verticalLayout_R.addWidget(optimize_Z_TB)
        self.verticalLayout_C.addWidget(imprinting_TB)
        self.verticalLayout_R.addWidget(periodicity_TB)
        if self.JPEG_GUI:
            self.verticalLayout_R.addWidget(HSV_TB)
        self.verticalLayout_R.addWidget(uniform_Z_control_TB)


        # Menu bar - removed:
        # self.menuBar = QtWidgets.QMenuBar(MainWindow)
        # self.menuBar.setGeometry(QtCore.QRect(0, 0, 549, 22))
        # self.menuBar.setObjectName("menuBar")
        # self.menuFIle = QtWidgets.QMenu(self.menuBar)
        # self.menuFIle.setObjectName("menuFIle")
        # self.menuEdit = QtWidgets.QMenu(self.menuBar)
        # self.menuEdit.setObjectName("menuEdit")
        # self.menuImage = QtWidgets.QMenu(self.menuBar)
        # self.menuImage.setObjectName("menuImage")
        # self.menuHelp = QtWidgets.QMenu(self.menuBar)
        # self.menuHelp.setObjectName("menuHelp")
        # MainWindow.setMenuBar(self.menuBar)
        # self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.menuFIle.setTitle(_translate("MainWindow", "FIle"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuImage.setTitle(_translate("MainWindow", "Image"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))

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

