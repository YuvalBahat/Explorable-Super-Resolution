from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from MainWindow import Ui_MainWindow
import os
import argparse

# Explorable SR imports:
from models import create_model
import options.options as option
import utils.util as util
from Z_optimization import Z_optimizer,ReturnPatchExtractionMat
from utils.logger import Logger
import data.util as data_util
import numpy as np
import torch
import qimage2ndarray
import cv2
import imageio
from skimage.transform import resize
from scipy.signal import find_peaks
import time
from collections import deque

# Explorable SR imports:
from KernelGAN import train as KernelGAN
from CEM.imresize_CEM import imresize

# Explorable JPEG imports:
from data.util import rgb2ycbcr
from JPEG_module.JPEG import JPEG
from skimage.color import rgb2hsv,hsv2rgb
from PIL import Image as PIL_Image
from jpeg2dct.numpy import load as jpeg_load

# General parameters:
Z_HISTORY_LENGTH = 10
IMAGE_FILE_EXST_FILTER = "PNG image files (*.png);; JPEG image files (*.jpg);;BMP image files (*.bmp)"

# Display options:
DISPLAY_ZOOM_FACTOR = 1
DISPLAY_ZOOM_FACTORS_RANGE = [1,4]
DISPLAY_INDUCED_LR = False
HORIZ_MAINWINDOW_OFFSET = 67+15 #67+5#When initializing the GUI, the OS horizontally shifts the main window by this amount compared to the desired set position. This value can be modified in case it varies from one OS to another, affecting the intial image canvas location.
ERR_MESSAGE_DURATION = 1e4
INFO_MESSAGE_DURATION = 5e3

# Optimization over Z signals space:
Z_OPTIMIZER_INITIAL_LR = 1e-1
Z_OPTIMIZER_INITIAL_LR_4_RANDOM = 1e-1
ONLY_MODIFY_MASKED_AREA_WHEN_OPTIMIZING = False
ITERS_PER_OPT_ROUND = 5
HIGH_OPT_ITERS_LIMIT = True
MARGINS_AROUND_REGION_OF_INTEREST = 30
RANDOM_OPT_INITS = False
MULTIPLE_OPT_INITS = False # When True, using random initializations in Z space in addition to the initialization with the current Z, when performing optimization in Z space (when using the variance, imprinting and all other Z optimization tools).
LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS = True
TRACK_ADDED_SCRIBLES = True # When True, using the recent modification that allows modifying drawing locations like imprints can be moved, rotated or resized.
Z_OPTIMIZATION_TIME_LIMIT = 30  # seconds
NON_LOCAL_Z_OPTIMIZATION = True #When True, optimizing over the entire region passed to the optimizer (depends on MARGINS_AROUND_REGION_OF_INTEREST), while penalizing for changes in the masked regions.
AUTO_MASK_GRAPHICAL_INPUT = True and NON_LOCAL_Z_OPTIMIZATION#When NON_LOCAL_Z_OPTIMIZATION is on, would automatically surround scribble mask by dilation to automatically create mask.

# For the "desired dictionary of patches" tool, allowing multi-scale dictionary:
DOWNSCALED_HIST_VERSIONS = False#0.9
MIN_DOWNSCALING_4_HIST = 0.75
DICTIONARY_REPLACES_HISTOGRAM = True
L1_REPLACES_HISTOGRAM = False
NO_DC_IN_PATCH_HISTOGRAM = True

# The random diverse alternatives tool:
NUM_RANDOM_ZS = 1
VGG_RANDOM_DOMAIN = False

# The variance manipulation tool:
RELATIVE_STD_OPT = True
LOCAL_STD_4_OPT = True

# The adversary tool:
D_EXPECTED_LR_SIZE = 64

# Periodicity encouraging tool:
AUTO_CYCLE_LENGTH_4_PERIODICITY = True

# SR-only parameters:
DISPLAY_ESRGAN_RESULTS = True

# JPEG-only tools:
MAXIMAL_JPEG_QF = 49

assert not (DICTIONARY_REPLACES_HISTOGRAM and L1_REPLACES_HISTOGRAM)

# Technical defenitions:
SCRIBBLE_MODES = ['pencil','line', 'polygon','ellipse', 'rect','imprinting','imprinting_auto_location']
MODES = ['selectpoly', 'selectrect','indicatePeriodicity','dropper',]+SCRIBBLE_MODES
LOCAL_TV_MASK_IDENTIFIERS_RANGE = [3,50]
IMPRINT_SIZE_CHANGES = ['narrower','wider','taller','shorter']
IMPRINT_LOCATION_CHANGES = ['left','right','up','down']
IMPRINT_ROTATIONS = ['clockwise','counter_clockwise']
CANVAS_DIMENSIONS = 600, 400
SELECTION_PEN = QPen(QColor(0xff, 0x00, 0x00), 3, Qt.DashLine)
PREVIEW_PEN = QPen(QColor(0xff, 0xff, 0xff), 1, Qt.SolidLine)

class Canvas(QLabel):

    mode = 'rectangle'
    secondary_color = QColor(Qt.white)

    primary_color_updated = pyqtSignal(str)
    secondary_color_updated = pyqtSignal(str)

    # Store configuration settings:
    config = {
        # Drawing options.
        'size': 1,
        'fill': True,
    }

    active_color = None
    preview_pen = None
    timer_event = None
    existing_selection_timer_event = None
    selection_display = None

    def initialize(self):
        self.background_color = QColor(self.secondary_color) if self.secondary_color else QColor(Qt.white)
        self.reset()

    def reset(self,canvas_dimensions=CANVAS_DIMENSIONS):
        # Create the pixmap for display.
        self.setPixmap(QPixmap(*canvas_dimensions))

        # Clear the canvas.
        self.pixmap().fill(self.background_color)

    def set_primary_color(self, hex):
        self.primary_color = QColor(hex)
        self.color_button.setStyleSheet('QPushButton { background-color: %s; }' % hex)
        self.color_state = 0
        transparent_icon = QIcon()
        transparent_icon.addPixmap(QPixmap("icons/transparent.png"), QIcon.Normal, QIcon.Off)
        self.color_button.setIcon(transparent_icon)

    def cycle_color_state(self):
        self.color_state = np.mod(self.color_state+1,4)
        if self.color_state==0:
            transparent_icon = QIcon()
            transparent_icon.addPixmap(QPixmap("icons/transparent.png"), QIcon.Normal, QIcon.Off)
            self.color_button.setIcon(transparent_icon)
            # self.color_button.setStyleSheet('QPush_button { background-color: %s; }' % self.primary_color.name())
        elif self.color_state==1:
            brightness_up_icon = QIcon()
            brightness_up_icon.addPixmap(QPixmap("icons/brightness_increase.png"), QIcon.Normal, QIcon.Off)
            self.color_button.setIcon(brightness_up_icon)
        elif self.color_state == 2:
            brightness_down_icon = QIcon()
            brightness_down_icon.addPixmap(QPixmap("icons/brightness_decrease.png"), QIcon.Normal,QIcon.Off)
            self.color_button.setIcon(brightness_down_icon)
        elif self.color_state == 3:
            brightness_down_icon = QIcon()
            brightness_down_icon.addPixmap(QPixmap("icons/fixed_color.png"), QIcon.Normal, QIcon.Off)
            self.color_button.setIcon(brightness_down_icon)

    def Scribble_Color(self):
        if self.color_state==0: #Normal scribble
            return self.primary_color
        else:
            if time.time()-self.latest_scribble_color_reset>3:
                self.cyclic_color_shift = 0
            else:
                self.cyclic_color_shift = np.mod(self.cyclic_color_shift + 20, 255)
            self.latest_scribble_color_reset = time.time()
            if self.color_state==1: # Brightness increase
                return QColor(255,self.cyclic_color_shift,self.cyclic_color_shift)
            elif self.color_state==2: # Brightness decrease
                return QColor(self.cyclic_color_shift,self.cyclic_color_shift,255)
            else: # Local Total Variations:
                return QColor(np.mod(self.cyclic_color_shift,255), 255,np.mod(50*self.local_TV_identifier,255))

    def set_config(self, key, value):
        self.config[key] = value

    def set_mode(self, mode):
        # Clean up active timer animations.
        self.timer_cleanup()
        if mode in ['selectpoly','selectrect']:
            self.selection_display_timer_cleanup(clear_data=True)
        # Reset mode-specific vars (all)
        self.active_shape_fn = None
        self.active_shape_args = ()

        self.origin_pos = None
        self.current_pos = None
        self.last_pos = None
        self.history_pos = None
        self.last_history = []
        self.current_text = ""
        self.last_text = ""
        self.last_config = {}
        self.dash_offset = 0
        self.locked = False
        # Apply the mode
        self.mode = mode

    def reset_mode(self):
        self.set_mode(self.mode)

    def on_timer(self):
        if self.timer_event:
            self.timer_event()
        if self.existing_selection_timer_event:
            self.existing_selection_timer_event()

    def timer_cleanup(self,display_selection=False):
        if self.timer_event:
            # Stop the timer, then trigger cleanup.
            timer_event = self.timer_event
            self.timer_event = None
            timer_event(final=True)
            if display_selection:
                self.selection_display = self.active_shape_fn+''
                self.selection_display_timer_creation()

    def selection_display_timer_cleanup(self,clear_data=False):
        if self.existing_selection_timer_event:
            timer_event = self.existing_selection_timer_event
            self.existing_selection_timer_event = None
            timer_event(only_erase_exisitng=True)
        if clear_data:
            self.selection_display = None

    def selection_display_timer_creation(self):
        def to_rescaled_Qpoint(input):
            return QPoint(*[np.round(self.display_zoom_factor*self.H_L_domains_ratio*v).astype(int) for v in input])

        if self.selection_display == 'drawRect':
            selection_fn_arg = [QRect(to_rescaled_Qpoint(self.LR_mask_vertices[0]),to_rescaled_Qpoint(self.LR_mask_vertices[1]))]
        else: #Drawing a polygon:
            selection_fn_arg = [to_rescaled_Qpoint(p) for p in self.LR_mask_vertices]
        self.existing_selection_timer_event =\
            lambda only_erase_exisitng=False: self.selection_display_timerEvent(selection_display_fn=self.selection_display,
                selection_fn_arg=selection_fn_arg, only_erase_exisitng=only_erase_exisitng)
        self.existing_selection_timer_event(only_erase_exisitng=True)

    def selection_display_timerEvent(self,selection_display_fn,selection_fn_arg,only_erase_exisitng):
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        pen.setDashOffset(self.dash_offset)
        p.setPen(pen)
        #This cleans the exisitng selection rectangle because it performs XOR with it:
        getattr(p, selection_display_fn)(*selection_fn_arg)
        if not only_erase_exisitng: #This repaints the selection rectangle while selection process is not done:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, selection_display_fn)(*selection_fn_arg)
        self.update()

    def Add_scribble_2_Undo_list(self):
        # History scribble and scribble mask are saved for the entire image (ignoring Z_mask issues), in the original display dimensions. This means there is no downscaling and then upscaling back when undoing.
        self.scribble_history.append(qimage2ndarray.rgb_view(self.pixmap().toImage()))
        self.scribble_mask_history.append(qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:, :, 0])
        self.undo_scribble_button.setEnabled(True)

    def Undo_scribble(self,add_2_redo_list=True):
        # Assigning saved scribble to scrible image:
        self.image_4_scribbling_display_size = self.scribble_history.pop()
        saved_size_different_than_current = list(self.image_4_scribbling_display_size.shape[:2])!=[self.display_zoom_factor*v for v in self.HR_size]# Handling the case when saved scribble display size doesn't match current display size
        if saved_size_different_than_current:
            self.image_4_scribbling_display_size = util.ResizeScribbleImage(self.image_4_scribbling_display_size,dsize=tuple([self.display_zoom_factor*v for v in self.HR_size]))
        if add_2_redo_list:# Adding current scribble to redo list:
            display_index_2_return_2 = 1 * self.current_display_index
            if self.current_display_index!=self.scribble_display_index:
                self.SelectImage2Display(self.scribble_display_index)
            self.scribble_redo_list.append(qimage2ndarray.rgb_view(self.pixmap().toImage()))
            self.scribble_mask_redo_list.append(qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:, :, 0])
            self.redo_scribble_button.setEnabled(True)
            if display_index_2_return_2!=self.scribble_display_index:
                self.SelectImage2Display(display_index_2_return_2)
            self.imprinting_arrows_enabling(False)#When called from within finalize_imprinting in the process of modifying the imprint, the add_2_redo_list flag is set to False.
            # I therefore use this flag to tell when this is not the case, and disable imprinting modifications when imprinting (scribble) is manually undone. Currently this means that if imprinting is undone and redone, it can no longer be modified.
        # Assigning saved scribble mask canvas to scribble mask canvas itself:
        pixmap = QPixmap()
        saved_scribble_mask = self.scribble_mask_history.pop()
        if saved_size_different_than_current:
            saved_scribble_mask = util.ResizeCategorialImage(saved_scribble_mask,dsize=tuple([self.display_zoom_factor*v for v in self.HR_size]))
        pixmap_image = qimage2ndarray.array2qimage(saved_scribble_mask)
        pixmap.convertFromImage(pixmap_image)
        self.scribble_mask_canvas.setPixmap(pixmap)

        self.undo_scribble_button.setEnabled(len(self.scribble_history) > 0)
        self.Update_Image_Display()

    def Redo_scribble(self):
        display_index_2_return_2 = 1 * self.current_display_index
        if self.current_display_index!=self.scribble_display_index:
            self.SelectImage2Display(self.scribble_display_index)
        self.Add_scribble_2_Undo_list()
        if display_index_2_return_2!=self.scribble_display_index:
            self.SelectImage2Display(display_index_2_return_2)
        self.image_4_scribbling_display_size = self.scribble_redo_list.pop()
        saved_size_different_than_current = list(self.image_4_scribbling_display_size.shape[:2])!=[self.display_zoom_factor*v for v in self.HR_size]# Handling the case when saved scribble display size doesn't match current display size
        if saved_size_different_than_current:
            self.image_4_scribbling_display_size = util.ResizeScribbleImage(self.image_4_scribbling_display_size,dsize=tuple([self.display_zoom_factor*v for v in self.HR_size]))

        # Assigning saved scribble mask canvas to scribble mask canvas itself:
        pixmap = QPixmap()
        saved_scribble_mask = self.scribble_mask_redo_list.pop()
        if saved_size_different_than_current:
            saved_scribble_mask = util.ResizeCategorialImage(saved_scribble_mask,dsize=tuple([self.display_zoom_factor*v for v in self.HR_size]))
        pixmap_image = qimage2ndarray.array2qimage(saved_scribble_mask)
        pixmap.convertFromImage(pixmap_image)
        self.scribble_mask_canvas.setPixmap(pixmap)
        self.redo_scribble_button.setEnabled(len(self.scribble_redo_list) > 0)
        self.Update_Image_Display()

    # Mouse events.

    def mousePressEvent(self, e):
        if (self.mode in self.scribble_modes) and not self.within_drawing and not self.in_picking_desired_hist_mode:
            # self.scribble_mode_entry_operations(imprinting='imprint' in self.mode)
            self.scribble_mode_entry_operations(drawing='imprint' not in self.mode and self.color_state==0)
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

    def scribble_mode_entry_operations(self,drawing=True):
        self.Z_optimizer_Reset()
        self.SelectImage2Display(self.scribble_display_index)
        self.Add_scribble_2_Undo_list()
        self.imprinting_arrows_enabling(False)
        if self.color_state == 3:  # Advancing the local_TV_identifier cycle by 1:
            self.local_TV_identifier = np.mod(self.local_TV_identifier + 1 - LOCAL_TV_MASK_IDENTIFIERS_RANGE[0],np.diff(LOCAL_TV_MASK_IDENTIFIERS_RANGE)[0]) + \
                                       LOCAL_TV_MASK_IDENTIFIERS_RANGE[0]
        if TRACK_ADDED_SCRIBLES and drawing:
            self.saved_scribble_canvas = 1*qimage2ndarray.rgb_view(self.pixmap().toImage())
            self.saved_scribble_mask_canvas = 1*qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())
            self.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(np.zeros_like(self.saved_scribble_mask_canvas))))

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def any_scribbles_within_mask(self):
        return np.any(self.HR_mask_display_size*qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:, :, 0])

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            returnable =  fn(e)
            if self.mode in self.scribble_modes and not self.within_drawing:
                self.scribble_mode_exit_operations(drawing='imprint' not in self.mode and self.color_state==0)
                # self.scribble_mode_exit_operations(enable_loc_modification='imprint' not in self.mode,consistently_embed=self.color_state==0)
            return returnable

    def scribble_mode_exit_operations(self,drawing=True):
    # def scribble_mode_exit_operations(self, enable_loc_modification=True,consistently_embed=True):
        # Input would only be consistently embedded if both enable_loc_modification and consistently_embed are set to True
        if TRACK_ADDED_SCRIBLES and drawing:
        # if TRACK_ADDED_SCRIBLES and enable_loc_modification:
            self.desired_graphic_input_rotation = 0
            self.desired_graphic_input = (qimage2ndarray.rgb_view(self.pixmap().toImage())/255).astype(np.float32)
            self.desired_im_HR_mask_4_imprinting = qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:,:,0]
            if self.display_zoom_factor!=1:
                self.desired_graphic_input = util.ResizeScribbleImage(self.desired_graphic_input,dsize=tuple(self.HR_size))
                self.desired_im_HR_mask_4_imprinting = util.ResizeCategorialImage(self.desired_im_HR_mask_4_imprinting,
                    dsize=tuple([s // self.display_zoom_factor for s in self.desired_im_HR_mask_4_imprinting.shape[:2]]), inclusive=True)

            self.imprint_desired_mask_bounding_rect = np.array(cv2.boundingRect(np.argwhere(self.desired_im_HR_mask_4_imprinting)))
            self.top_left_corner = 1*self.imprint_desired_mask_bounding_rect[:2]
            self.target_imprinting_dimensions = 1*self.imprint_desired_mask_bounding_rect[2:]
            # Converting to the [x,y,widht,height] format used later:
            self.imprint_desired_mask_bounding_rect = self.imprint_desired_mask_bounding_rect[list([1,0,3,2])]
            self.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(self.saved_scribble_mask_canvas)))
            self.setPixmap(QPixmap(qimage2ndarray.array2qimage(self.saved_scribble_canvas)))
            if True: #Currently no supporting modifying location of non-drawing scribble tools (brightness manipulation and local-TV minimization), so I commented out unfinished code ment to support it.
            # if consistently_embed:
                self.embed_graphical_input(auto_location=False and self.special_behavior_button.isChecked())
            else:
                merged_mask = 1*self.saved_scribble_mask_canvas[:,:,0]
                added_locations = (1*self.desired_im_HR_mask_4_imprinting).astype(bool)
                merged_mask[added_locations] = 1*self.desired_im_HR_mask_4_imprinting[added_locations]
                self.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(np.tile(np.expand_dims(merged_mask,-1),[1,1,3]))))
                merged_canvas = 1*self.saved_scribble_canvas
                # added_locations = np.tile(np.expand_dims(added_locations, -1), [1, 1, 3])
                merged_canvas[added_locations] = 255*self.desired_graphic_input[added_locations]
                self.setPixmap(QPixmap(qimage2ndarray.array2qimage(self.saved_scribble_canvas)))
                self.imprinting_arrows_enabling(True)
                self.reset_mode()
        self.safe_scribble_apply_buttons_enabling()

    def safe_scribble_apply_buttons_enabling(self):
        any_scribble_within_mask = self.any_scribbles_within_mask()
        self.apply_scribble_button.setEnabled(any_scribble_within_mask)
        self.loop_apply_scribble_button.setEnabled(any_scribble_within_mask)


    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            returnable =  fn(e)
            if self.mode in self.scribble_modes and not self.within_drawing:
                self.scribble_mode_exit_operations(drawing='imprint' not in self.mode and self.color_state==0)
                # self.scribble_mode_exit_operations(enable_loc_modification='imprint' not in self.mode,consistently_embed=self.color_state==0)
            return returnable

    # Generic events (shared by brush-like tools)

    def generic_mousePressEvent(self, e):
        self.last_pos = e.pos()

        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color

    def generic_mouseReleaseEvent(self, e):
        self.last_pos = None

    def selectpoly_mousePressEvent(self, e):
        if not self.locked or e.button == Qt.RightButton:
            self.active_shape_fn = 'drawPolygon'
            self.preview_pen = SELECTION_PEN
            if self.history_pos is None:
                self.Avoid_Scribble_Display(True)
            self.generic_poly_mousePressEvent(e)

    def selectpoly_timerEvent(self, final=False):
        self.generic_poly_timerEvent(final)

    def selectpoly_mouseMoveEvent(self, e):
        if not self.locked:
            self.generic_poly_mouseMoveEvent(e)

    def selectpoly_mouseDoubleClickEvent(self, e):
        self.current_pos = e.pos()
        self.locked = True
        self.HR_selected_mask = np.zeros(self.HR_size)
        self.LR_mask_vertices = [(p.x()/self.H_L_domains_ratio/self.display_zoom_factor,p.y()/self.H_L_domains_ratio/self.display_zoom_factor)
                                 for p in (self.history_pos + [self.current_pos])]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        # I used to use HR mask that is pixel-algined in the LR domain, now changed to make sure it is pixel-aligned only in the HR domain (avoiding subpixel shifts due to self.display_zoom_factor):
        # self.HR_mask_vertices = [(coord[0]*self.H_L_domains_ratio,coord[1]*self.H_L_domains_ratio) for coord in self.LR_mask_vertices]
        if self.in_picking_desired_hist_mode and self.using_external_image:
            self.HR_mask_vertices = [(int(np.round(p.x())), int(np.round(p.y()))) for p in (self.history_pos + [self.current_pos])]
        else:
            self.HR_mask_vertices = [(int(np.round(p.x()/self.display_zoom_factor)),int(np.round(p.y()/self.display_zoom_factor)))
                                     for p in (self.history_pos + [self.current_pos])]
        self.HR_selected_mask = cv2.fillPoly(self.HR_selected_mask,[np.array(self.HR_mask_vertices)],(1,1,1))
        if not self.in_picking_desired_hist_mode:
            self.Z_mask = np.zeros(self.Z_size)
            if self.HR_Z:
                self.Z_mask = cv2.fillPoly(self.Z_mask, [np.array(self.HR_mask_vertices)], (1, 1, 1))
            else:
                self.Z_mask = cv2.fillPoly(self.Z_mask,[np.round(np.array(self.LR_mask_vertices)).astype(int)],(1,1,1))
            self.update_HR_mask_display_size()
            self.Update_Z_Sliders()
            self.Z_optimizer_Reset()
        self.selectpoly_button.setChecked(False)
        self.timer_cleanup(display_selection=True)
        self.Avoid_Scribble_Display(False)
        if not self.in_picking_desired_hist_mode:
            self.safe_scribble_apply_buttons_enabling()

    def update_HR_mask_display_size(self):
        self.HR_mask_display_size = util.ResizeCategorialImage(self.HR_selected_mask.astype(np.int16),
            dsize=tuple([self.display_zoom_factor*val for val in self.HR_size])).astype(self.HR_selected_mask.dtype)
        if self.JPEG_GUI:
            self.HR_mask_smeared_2_blocks_display_size = util.SmearMask2JpegBlocks(self.HR_mask_display_size,block_size=8*self.display_zoom_factor)

    def indicatePeriodicity_mousePressEvent(self, e):
        if not self.locked or e.button == Qt.RightButton:
            self.active_shape_fn = 'drawPolygon'
            self.preview_pen = SELECTION_PEN
            if self.history_pos is None:# Just started periodicity indication mode:
                self.Avoid_Scribble_Display(True)
            self.generic_poly_mousePressEvent(e)
        if len(self.history_pos)==1:
            self.statusBar.showMessage('Select a point in the desired 1st relative direction')
        elif len(self.history_pos)==2:
            self.statusBar.showMessage('Select a point in the desired 2nd relative direction')
        elif len(self.history_pos)==3:
            self.locked = True
            self.IncreasePeriodicity_2D_button.setEnabled(True)
            self.IncreasePeriodicity_1D_button.setEnabled(True)
            self.Z_optimizer_Reset()
            if AUTO_CYCLE_LENGTH_4_PERIODICITY:
                def im_coordinates_2_grid(points):
                    points = [(p.y(),p.x()) for p in points]
                    grid = []
                    num_steps = int(max([np.abs(points[1][axis]-points[0][axis]) for axis in range(2)])/0.1)
                    for axis in range(2):
                        grid.append(np.linspace(start=points[0][axis]/self.HR_size[axis]*2-1,stop=points[1][axis]/self.HR_size[axis]*2-1,num=num_steps))
                    return np.reshape(grid[::-1],[2,-1]).transpose((1,0)) #Reversing the order of axis (the grid list) because grid_sample expects (x,y) coordinates
                def autocorr(x):
                    x -= x.mean()
                    result = np.correlate(x, x, mode='full')
                    normalizer = [val+1 for val in range(len(x))]
                    normalizer = np.array(normalizer+normalizer[-2::-1])
                    result /= normalizer
                    return result[x.size:]
                def line_length(points):
                    points = [np.array([p.y(),p.x()]) for p in points]
                    return np.linalg.norm(points[1]-points[0])

                self.periodicity_points = []
                for p in self.history_pos[1:]:
                    image_along_line = torch.nn.functional.grid_sample(torch.mean(self.random_Z_images[0],dim=0,keepdim=True).unsqueeze(0),
                        torch.from_numpy(im_coordinates_2_grid([self.history_pos[0],p])).view([1,1,-1,2]).to(self.random_Z_images.device).type(self.random_Z_images.dtype)).squeeze().data.cpu().numpy()
                    autocorr_peaks = find_peaks(autocorr(image_along_line))[0]
                    cur_point = np.array([p.y()-self.history_pos[0].y(),p.x()-self.history_pos[0].x()])
                    if len(autocorr_peaks)>0:
                        autocorr_peaks = [peak for peak in autocorr_peaks if autocorr(image_along_line)[peak]>1e-3]
                        if len(autocorr_peaks) > 0:
                            cur_length = line_length([self.history_pos[0], p])
                            step_size = cur_length / image_along_line.size
                            cycle_length = step_size*autocorr_peaks[0]
                            cur_point = cur_point / cur_length * cycle_length
                    print('Adding periodicity point (y,x) = (%.3f,%.3f)'%(cur_point[0],cur_point[1]))
                    self.periodicity_points.append(cur_point)
                self.periodicity_mag_1_button.setValue(np.linalg.norm(self.periodicity_points[0]))
                self.periodicity_mag_2_button.setValue(np.linalg.norm(self.periodicity_points[1]))
                self.statusBar.showMessage('Automatically evaluated current periodicity.', INFO_MESSAGE_DURATION)
            else:
                self.periodicity_points = [(p.y()-self.history_pos[0].y(),p.x()-self.history_pos[0].x()) for p in self.history_pos[1:]]
            SHOW_CHOSEN_POINTS = True
            if SHOW_CHOSEN_POINTS:
                self.timer_cleanup()
                p = QPainter(self.pixmap())
                p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
                p.setPen(QPen(QColor(0xff, 0x00, 0x00), 5, Qt.SolidLine))
                getattr(p, 'drawPoint')(self.history_pos[0].x(),self.history_pos[0].y())
                for point in self.periodicity_points:
                    getattr(p, 'drawPoint')(np.round(point[1])+self.history_pos[0].x(), np.round(point[0])+self.history_pos[0].y())

    def indicatePeriodicity_timerEvent(self, final=False):
        self.generic_poly_timerEvent(final)

    def indicatePeriodicity_mouseMoveEvent(self, e):
        if not self.locked:
            self.generic_poly_mouseMoveEvent(e)

    # Select rectangle events
    def Avoid_Scribble_Display(self,avoid_not_return):
        #Used for preventing the addition of dashed lines to the desired scribble, by exiting the scribble canvas and returning at the end
        if not self.in_picking_desired_hist_mode:
            if avoid_not_return:
                self.should_return_2_scribble_display = self.current_display_index==self.scribble_display_index
                if self.should_return_2_scribble_display:
                    self.SelectImage2Display(self.cur_Z_im_index)
            elif self.should_return_2_scribble_display:
                self.SelectImage2Display(self.scribble_display_index)

    def selectrect_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.preview_pen = SELECTION_PEN
        self.Avoid_Scribble_Display(True)
        self.generic_shape_mousePressEvent(e)

    def selectrect_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def selectrect_mouseMoveEvent(self, e):
        if not self.locked:
            self.current_pos = e.pos()

    def update_mask_bounding_rect(self):
        self.mask_bounding_rect = np.array(cv2.boundingRect(np.stack([np.round(np.array(p)).astype(int) for p in self.LR_mask_vertices], 1).transpose()))
        if not self.JPEG_GUI:
            self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.mask_bounding_rect[2:]]))
        self.contained_Z_mask = True

    def selectrect_mouseReleaseEvent(self, e):
        self.current_pos = e.pos()
        self.locked = True
        self.HR_selected_mask = np.zeros(self.HR_size)
        self.LR_mask_vertices = [(p.x()/self.H_L_domains_ratio/self.display_zoom_factor,p.y()/self.H_L_domains_ratio/self.display_zoom_factor) for p in [self.origin_pos, self.current_pos]]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        # I used to use HR mask that is pixel-algined in the LR domain, now changed to make sure it is pixel-aligned only in the HR domain (avoiding subpixel shifts due to self.display_zoom_factor):
        # self.HR_mask_vertices = [(coord[0]*self.H_L_domains_ratio,coord[1]*self.H_L_domains_ratio) for coord in self.LR_mask_vertices]
        if self.in_picking_desired_hist_mode and self.using_external_image:
            self.HR_mask_vertices = [(int(np.round(p.x())), int(np.round(p.y()))) for p in [self.origin_pos, self.current_pos]]
        else:
            self.HR_mask_vertices = [(int(np.round(p.x()/self.display_zoom_factor)),int(np.round(p.y()/self.display_zoom_factor))) for p in [self.origin_pos, self.current_pos]]
        self.HR_selected_mask = cv2.rectangle(self.HR_selected_mask,self.HR_mask_vertices[0],self.HR_mask_vertices[1],(1,1,1),cv2.FILLED)
        if not self.in_picking_desired_hist_mode:
            self.Z_mask = np.zeros(self.Z_size)
            if self.HR_Z:
                self.Z_mask = cv2.rectangle(self.Z_mask, self.HR_mask_vertices[0], self.HR_mask_vertices[1], (1, 1, 1),cv2.FILLED)
            else:
                self.Z_mask = cv2.rectangle(self.Z_mask,tuple(np.round(np.array(self.LR_mask_vertices[0])).astype(np.int)),
                                            tuple(np.round(np.array(self.LR_mask_vertices[1])).astype(np.int)),(1,1,1),cv2.FILLED)
            self.update_HR_mask_display_size()
            self.Update_Z_Sliders()
            self.Z_optimizer_Reset()
        self.selectrect_button.setChecked(False)#This does not work, probably because of some genral property set for all "mode" buttons.
        self.timer_cleanup(display_selection=True)
        self.Avoid_Scribble_Display(False)
        if not self.in_picking_desired_hist_mode:
            self.safe_scribble_apply_buttons_enabling()

    def Z_optimizer_Reset(self):
        self.Z_optimizer_initial_LR = Z_OPTIMIZER_INITIAL_LR
        self.Z_optimizer = None
        self.Z_optimizer_logger = None

    def ReturnMaskedMapAverage(self,map):
        return np.sum(map*self.Z_mask)/np.sum(self.Z_mask)

    def Update_Z_Sliders(self):
        z0_slider_new_val = self.ReturnMaskedMapAverage(self.control_values[0].data.cpu().numpy())
        Z1_slider_new_val = self.ReturnMaskedMapAverage(self.control_values[1].data.cpu().numpy())
        third_channel_slider_new_val = self.ReturnMaskedMapAverage(self.control_values[2].data.cpu().numpy())
        self.Z0_slider.setSliderPosition(100*z0_slider_new_val)
        self.Z1_slider.setSliderPosition(100*Z1_slider_new_val)
        self.third_channel_slider.setSliderPosition(100*third_channel_slider_new_val)
        self.previous_sliders_values = np.expand_dims(self.Z_mask,0)*np.array([z0_slider_new_val,Z1_slider_new_val,third_channel_slider_new_val]).reshape([3,1,1])+\
                                       np.expand_dims(1-self.Z_mask, 0)*self.previous_sliders_values

    # Pencil events

    def pencil_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)
        self.pencil_mouseMoveEvent(e)

    def pencil_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), self.config['size']*self.display_zoom_factor, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            scribble_mask = QPainter(self.scribble_mask_canvas.pixmap())
            scribble_mask.setPen(QPen(self.Scribble_Mask_Color(), self.config['size']*self.display_zoom_factor, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            scribble_mask.drawLine(self.last_pos, e.pos())
            self.scribble_mask_canvas.update()
            self.last_pos = e.pos()
            self.update()

    def Scribble_Mask_Color(self):
        if self.color_state in [0,1,2]:
            return QColor(self.color_state+1,self.color_state+1,self.color_state+1)
        else: #Local TV
            return QColor(self.local_TV_identifier+1,self.local_TV_identifier+1,self.local_TV_identifier+1)

    def pencil_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Dropper events

    def dropper_mousePressEvent(self, e):
        c = self.pixmap().toImage().pixel(e.pos())
        hex = QColor(c).name()

        if e.button() == Qt.LeftButton:
            self.set_primary_color(hex)
            self.primary_color_updated.emit(hex)  # Update UI.

    # Generic shape events: Rectangle, Ellipse, Rounded-rect
    def generic_shape_mousePressEvent(self, e):
        self.origin_pos = e.pos()
        self.current_pos = e.pos()
        self.timer_event = self.generic_shape_timerEvent

    def generic_shape_timerEvent(self, final=False):
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        pen.setDashOffset(self.dash_offset)
        p.setPen(pen)
        if self.last_pos: #This cleans the exisitng selection rectangle because it performs XOR with it:
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.last_pos), *self.active_shape_args)

        if not final: #This repaints the selection rectangle while selection process is not done:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.current_pos), *self.active_shape_args)

        self.update()
        self.last_pos = self.current_pos

    def generic_shape_mouseMoveEvent(self, e):
        self.current_pos = e.pos()

    def generic_shape_mouseReleaseEvent(self, e):
        if self.last_pos:
            # Clear up indicator.
            self.timer_cleanup()
            line_width = 1 if self.config['fill'] else self.config['size']*self.display_zoom_factor
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), line_width, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
            scribble_mask = QPainter(self.scribble_mask_canvas.pixmap())
            scribble_mask.setPen(QPen(self.Scribble_Mask_Color(), line_width, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))

            if self.config['fill']:
                p.setBrush(QBrush(self.Scribble_Color()))
                scribble_mask.setBrush(QBrush(self.Scribble_Mask_Color()))
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
            getattr(scribble_mask, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
            self.scribble_mask_canvas.update()
            self.update()

        self.reset_mode()

    # Line events

    def line_mousePressEvent(self, e):
        self.origin_pos = e.pos()
        self.current_pos = e.pos()
        self.preview_pen = QPen(PREVIEW_PEN)
        self.preview_pen.setWidth(self.config['size']*self.display_zoom_factor)
        self.timer_event = self.line_timerEvent

    def line_timerEvent(self, final=False):
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        p.setPen(pen)
        if self.last_pos:
            p.drawLine(self.origin_pos, self.last_pos)

        if not final:
            p.drawLine(self.origin_pos, self.current_pos)

        self.update()
        self.last_pos = self.current_pos

    def line_mouseMoveEvent(self, e):
        self.current_pos = e.pos()

    def line_mouseReleaseEvent(self, e):
        if self.last_pos:
            # Clear up indicator.
            self.timer_cleanup()

            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), self.config['size']*self.display_zoom_factor, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            p.drawLine(self.origin_pos, e.pos())
            scribble_mask = QPainter(self.scribble_mask_canvas.pixmap())
            scribble_mask.setPen(QPen(self.Scribble_Mask_Color(), self.config['size']*self.display_zoom_factor, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            scribble_mask.drawLine(self.origin_pos, e.pos())
            self.scribble_mask_canvas.update()
            self.update()

        self.reset_mode()

    # Generic poly events
    def generic_poly_mousePressEvent(self, e):
        SHOW_CHOSEN_POINTS = False
        self.within_drawing = self.mode in self.scribble_modes # I add this to distinguish between first mouse press (initiating polygon drawing) and the rest of the presses. The motivation is to know when I BEGIN a scribble action.
        if SHOW_CHOSEN_POINTS and self.mode=='indicatePeriodicity' and self.locked:
            self.timer_cleanup()
            self.timer_event = self.indicatePeriodicity_timerEvent
        else:
            if e.button() == Qt.LeftButton:
                if self.history_pos:
                    self.history_pos.append(e.pos())
                else:
                    self.history_pos = [e.pos()]
                    self.current_pos = e.pos()
                    self.timer_event = self.generic_poly_timerEvent

            elif e.button() == Qt.RightButton and self.history_pos:
                # Clean up, we're not drawing
                self.timer_cleanup()
                self.reset_mode()

    def indicatePeriodicity_timerEvent(self, final=False):
        periodicity_points_pen = QPen(QColor(0xff, 0x00, 0x00), 5, Qt.SolidLine)
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        p.setPen(periodicity_points_pen)
        getattr(p, 'drawPoint')(self.history_pos[0].x(), self.history_pos[0].y())
        for point in self.periodicity_points:
            getattr(p, 'drawPoint')(np.round(point[1]) + self.history_pos[0].x(),
                                    np.round(point[0]) + self.history_pos[0].y())
        self.update()

    def generic_poly_timerEvent(self, final=False):
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        pen.setDashOffset(self.dash_offset)
        p.setPen(pen)
        if self.last_history:
            getattr(p, self.active_shape_fn)(*self.last_history)

        if not final:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, self.active_shape_fn)(*self.history_pos + [self.current_pos])

        self.update()
        self.last_pos = self.current_pos
        self.last_history = self.history_pos + [self.current_pos]

    def generic_poly_mouseMoveEvent(self, e):
        self.current_pos = e.pos()

    def generic_poly_mouseDoubleClickEvent(self, e):
        self.within_drawing = False # I add this to distinguish between first mouse press (initiating polygon drawing) and the rest of the presses. The motivation is to know when I BEGIN a scribble action.
        self.timer_cleanup()
        p = QPainter(self.pixmap())
        line_width = 1 if self.config['fill'] else self.config['size'] * self.display_zoom_factor
        p.setPen(QPen(self.Scribble_Color(), line_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        scribble_mask = QPainter(self.scribble_mask_canvas.pixmap())
        scribble_mask.setPen(QPen(self.Scribble_Mask_Color(), line_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        # Note the brush is ignored for polylines.
        if self.config['fill']:
            p.setBrush(QBrush(self.Scribble_Color()))
            scribble_mask.setBrush(QBrush(self.Scribble_Mask_Color()))

        getattr(p, self.active_shape_fn)(*self.history_pos + [e.pos()])
        getattr(scribble_mask, self.active_shape_fn)(*self.history_pos + [e.pos()])
        self.update()
        self.scribble_mask_canvas.update()
        self.reset_mode()

    # Rectangle events
    def rect_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.active_shape_args = ()
        self.preview_pen = QPen(PREVIEW_PEN)
        self.generic_shape_mousePressEvent(e)

    def rect_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def rect_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def rect_mouseReleaseEvent(self, e):
        self.generic_shape_mouseReleaseEvent(e)

    # Input image scribble events:

    def imprinting_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.active_shape_args = ()
        self.preview_pen = SELECTION_PEN
        self.generic_shape_mousePressEvent(e)

    def imprinting_auto_location_mousePressEvent(self, e):
        self.imprinting_mousePressEvent(e)

    def imprinting_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def imprinting_auto_location_timerEvent(self, final=False):
        self.imprinting_timerEvent(final)

    def imprinting_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def imprinting_auto_location_mouseMoveEvent(self, e):
        self.imprinting_mouseMoveEvent(e)

    def imprinting_mouseReleaseEvent(self, e):
        self.imprinting_location_boundaries = None
        self.finalize_imprinting(e,transparent_mask=self.special_behavior_button.isChecked())

    def imprinting_auto_location_mouseReleaseEvent(self, e):
        if self.imprinting_location_boundaries is None:
            self.imprinting_location_boundaries = []
        self.finalize_imprinting(e,transparent_mask=self.special_behavior_button.isChecked())

    def FindOptimalImprintingLocation(self,desired_mask_bounding_rect):
        # For SR imprinting:
        NUM_BEST_2_KEEP = 4
        NUM_SAMPLES_IN_RANGE = 10*NUM_BEST_2_KEEP
        def crop_LR_im(cropping_location):
            return self.SR_model.model_input.data[0].cpu().numpy().transpose(1,2,0)[cropping_location[0]:cropping_location[2],cropping_location[1]:cropping_location[3]:,-3:]
        HR_im_projected_2_ortho_nullspace = self.Project_2_Orthog_Nullspace(self.output_image_0_1[0].data.cpu().numpy().transpose(1,2,0))
        def crop_HR_im(cropping_location):
            return HR_im_projected_2_ortho_nullspace[cropping_location[0]:cropping_location[2],cropping_location[1]:cropping_location[3]:,:]
        def return_average_abs_im_diff(existing_im_loc,desired_HR_im,desired_HR_im_mask,LR_phase):
            if LR_phase:
                existing_im = crop_LR_im(existing_im_loc)
                org_size = existing_im.shape[:2]
            else:
                org_size = np.array([existing_im_loc[2]-existing_im_loc[0],existing_im_loc[3]-existing_im_loc[1]])
                existing_im = crop_HR_im(existing_im_loc)
            resized_desired_im = util.ResizeScribbleImage(desired_HR_im,dsize=tuple([v*(self.opt['scale'] if LR_phase else 1) for v in org_size]))
            resized_desired_im_mask = \
                util.ResizeCategorialImage(desired_HR_im_mask.astype(np.uint8),dsize=tuple([v*(self.opt['scale'] if LR_phase else 1) for v in org_size])
                                           ,inclusive=True)
            if LR_phase:
                resized_desired_im = imresize(resized_desired_im,1/self.opt['scale'])
                resized_desired_im_mask = imresize(resized_desired_im_mask,1/self.opt['scale'])!=0
            return np.sum(np.abs(resized_desired_im-existing_im)*np.expand_dims(resized_desired_im_mask,-1))/np.sum(resized_desired_im_mask)/3

        desired_image = self.Project_2_Orthog_Nullspace(self.desired_graphic_input)[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1] + desired_mask_bounding_rect[3],
                        desired_mask_bounding_rect[0]:desired_mask_bounding_rect[0] + desired_mask_bounding_rect[2],...]
        desired_image_mask = self.desired_im_HR_mask_4_imprinting[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1] + desired_mask_bounding_rect[3],
                             desired_mask_bounding_rect[0]:desired_mask_bounding_rect[0] + desired_mask_bounding_rect[2]]
        original_boundaries = np.array([sorted([self.imprinting_location_boundaries[0][dim_num],self.imprinting_location_boundaries[1][dim_num]]) for dim_num in range(4)]).transpose()
        def keep_within_range(location):
            return np.array([np.minimum(np.maximum(location[dim_num],original_boundaries[0][dim_num]),original_boundaries[1][dim_num]) for dim_num in range(4)])
        self.imprinting_location_boundaries = [self.imprinting_location_boundaries] #Keeping NUM_BEST_2_KEEP boundary sets at each iteration, except for the first one in each phase (LR,HR),
        # where there will be only one set. So making this a sets list of length 1.
        initially_sampled_area = original_boundaries[1]-original_boundaries[0]
        initially_sampled_area = initially_sampled_area[0]*initially_sampled_area[1]+initially_sampled_area[2]*initially_sampled_area[3]
        samples_per_best_location = [np.maximum(NUM_SAMPLES_IN_RANGE,initially_sampled_area//20)]
        for LR_phase in [True,False]:
            latest_LR_diff = 1
            while True:
                sampled_locations = []
                for set_num,boundary_set in enumerate(self.imprinting_location_boundaries):
                    samples_2_add = np.zeros([samples_per_best_location[set_num], 4])
                    for dim_num in range(4):
                        invalid_samples = np.ones([samples_per_best_location[set_num]]).astype(np.bool)
                        dim_range = sorted([boundary_set[0][dim_num],boundary_set[1][dim_num]])
                        while np.any(invalid_samples):
                            samples_2_add[invalid_samples,dim_num] = np.random.randint(low=int(np.floor(dim_range[0] / (self.opt['scale'] if LR_phase else 1))),
                                high=1 + int(np.ceil(dim_range[1] / (self.opt['scale'] if LR_phase else 1))),size=[np.sum(invalid_samples)])
                            if dim_num>1:
                                invalid_samples = samples_2_add[:,dim_num]-samples_2_add[:,dim_num-2]<(1 if LR_phase else self.opt['scale'])
                            else:
                                invalid_samples = np.zeros_like(invalid_samples)

                    sampled_locations.append(samples_2_add.astype(np.int32))
                sampled_locations = np.concatenate(sampled_locations,0)
                average_LR_diffs = np.array([return_average_abs_im_diff(loc,desired_image,desired_image_mask,LR_phase=LR_phase) for loc in sampled_locations])
                leading_location_inds = np.argsort(average_LR_diffs)
                if average_LR_diffs[leading_location_inds[0]]>=latest_LR_diff:
                    if LR_phase:
                        half_range = int(np.ceil(self.opt['scale'] / 2))
                        self.imprinting_location_boundaries = [[keep_within_range(best_locations[0]*self.opt['scale'] + v) for v in [-half_range, half_range]]]
                        samples_per_best_location = [NUM_SAMPLES_IN_RANGE]
                    break
                latest_LR_diff = average_LR_diffs[leading_location_inds[0]]
                best_locations = sampled_locations[leading_location_inds[:NUM_BEST_2_KEEP]]
                self.imprinting_location_boundaries = [[keep_within_range((self.opt['scale'] if LR_phase else 1)*(best_location + v)) for v in [-1, 1]] for best_location in best_locations]
                # Distributing number of samples per chosen location according to the locations' scores:
                samples_per_best_location = 1/np.maximum(0.01,(average_LR_diffs[leading_location_inds[:NUM_BEST_2_KEEP]]))
                samples_per_best_location = (samples_per_best_location/np.sum(samples_per_best_location)*NUM_SAMPLES_IN_RANGE).astype(int)
                samples_per_best_location[-1] = NUM_SAMPLES_IN_RANGE-np.sum(samples_per_best_location[:-1])

        best_location = best_locations[0]
        self.target_imprinting_dimensions = np.array([np.abs(best_location[2] - best_location[0]) + 1, np.abs(best_location[3] - best_location[1]) + 1])
        self.top_left_corner = np.array([np.minimum(best_location[2], best_location[0]), np.minimum(best_location[3], best_location[1])])

    def Find_Optimal_Desired_Imprint_Mask(self,desired_mask_bounding_rect,fixed_image,desired_image,desired_im_mask,padding,scribble_mode=False,org_choice_rectangle=None):
        # For JPEG imprinting:
        # scribble_mode - Means I search across different rescalings of the desired input, rather than different cropping bounderies.
        # fixed_image & desired_image should both be in [0,255]
        SEARCH_IN_GRAYSCALE = True
        # desired_image = 255*self.desired_graphic_input
        if SEARCH_IN_GRAYSCALE:
            desired_image = rgb2ycbcr(1*desired_image)
            fixed_image = rgb2ycbcr(1*fixed_image)
        def crop_padding(array):
            return np.pad(array, (tuple(padding[0]), tuple(padding[1]))) if SEARCH_IN_GRAYSCALE else np.pad(array, (tuple(padding[0]), tuple(padding[1]), (0, 0)))

        fixed_im_coeffs = self.SR_model.jpeg_compressor_Y(torch.from_numpy(fixed_image).unsqueeze(0).unsqueeze(0))
        original_rectangle = 1 * desired_mask_bounding_rect
        if scribble_mode:
            originally_marked_rectangle_mask = np.zeros_like(desired_im_mask)
            originally_marked_rectangle_mask[org_choice_rectangle[1]:org_choice_rectangle[1] +org_choice_rectangle[3], \
                                                    org_choice_rectangle[0]:org_choice_rectangle[0] +org_choice_rectangle[2]] = 1
        else:
            originally_marked_rectangle_mask = 1 * desired_im_mask
        original_desired_portion_of_rectangle = originally_marked_rectangle_mask[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1]+desired_mask_bounding_rect[3],\
            desired_mask_bounding_rect[0]:desired_mask_bounding_rect[0]+desired_mask_bounding_rect[2]].mean()

        optional_rects = [desired_mask_bounding_rect]
        offset_grids = [[i for i in range(-7,9,3)],[-1,0,1]]
        self.statusBar.showMessage('Automatically fine-tuning desired imprint borders...')
        for scale_num in range(len(offset_grids)):
            def return_optional_coords(rect_coord):
                optional_coords = [[]]
                if rect_coord.size>1:
                    optional_coords = return_optional_coords(rect_coord[1:])
                return [[rect_coord[0]+i]+suffix for i in offset_grids[scale_num] for suffix in optional_coords]
            # Start by enumerating all possible shifts of each of the 4 coordinates:
            optional_desired_im_rect_coords = [return_optional_coords(rect) for rect in optional_rects]
            # Turn coordinate foursomes into np arrays:
            optional_desired_im_rect_coords = [np.array(coords_) for coords in optional_desired_im_rect_coords for coords_ in coords]
            # Discard foursomes that go beyond image boundaries or correspond to rectangles of size<4 (not sure why 4 pixels):
            optional_desired_im_rect_coords = \
                [coords for coords in optional_desired_im_rect_coords if (np.all(coords[:2]>=0) and \
                    np.all(coords[:2]+coords[2:]<=desired_image.shape[:2][::-1]) and np.all(coords[2:]>=4))]
            # Discard foursomes corresponding to regions with less than MIN_OVERLAP_WITH_MARKED_RECTANGLE in common with the originaly marked rectangle, where
            # MIN_OVERLAP_WITH_MARKED_RECTANGLE is first adjusted to account for non-rectangle masks, that cause the originally desired area to cover only part of the original rectangle.:
            MIN_OVERLAP_WITH_MARKED_RECTANGLE = 0.5
            MIN_OVERLAP_WITH_MARKED_RECTANGLE *= original_desired_portion_of_rectangle
            optional_desired_im_rect_coords = \
                [coords for coords in optional_desired_im_rect_coords if\
                 np.mean(util.crop_nd_array(originally_marked_rectangle_mask, coords))>=MIN_OVERLAP_WITH_MARKED_RECTANGLE]
            optional_crops = [util.crop_nd_array(desired_image,coords) for coords in optional_desired_im_rect_coords]
            optional_masks = [util.crop_nd_array(desired_im_mask,coords) for coords in optional_desired_im_rect_coords]
            optional_imprints = []
            for option_num in range(len(optional_crops)):
                crop = crop_padding(util.ResizeScribbleImage(optional_crops[option_num],dsize=tuple(self.target_imprinting_dimensions)))
                mask = crop_padding(util.ResizeCategorialImage(optional_masks[option_num].astype(np.uint8),dsize=tuple(self.target_imprinting_dimensions),inclusive=True))
                if not SEARCH_IN_GRAYSCALE:
                    mask = np.expand_dims(mask,-1)
                optional_imprints.append(crop*mask+fixed_image*(1-mask))
            calc_batches = 1
            success = False
            while not success: #Dealing with limited memory resources when calculating DCT coefficients of all possible imprints:
                optional_coeffs = []
                b_size = int(np.ceil(len(optional_imprints)/calc_batches))
                try:
                    for b_num in range(calc_batches):
                        optional_coeffs.append(self.SR_model.jpeg_compressor_Y_non_quantized(torch.from_numpy(np.stack(optional_imprints[b_num*b_size:(b_num+1)*b_size],0)).unsqueeze(1).to(self.SR_model.jpeg_compressor_Y.device)))
                    success = True
                except:
                    calc_batches += 1
                    if calc_batches>50:
                        raise Exception('Too many attempts, check what''s the problem.')
            optional_coeffs = torch.cat(optional_coeffs,0)
            best_coords_num = torch.max(torch.tensor(0).type(fixed_im_coeffs.type()),(optional_coeffs-fixed_im_coeffs).abs()-0.5).sum(-1).sum(-1).sum(-1).argsort()
            if scale_num==0:
                optional_rects = [optional_desired_im_rect_coords[i.item()] for i in best_coords_num[:10]]
            else:
                self.statusBar.showMessage('Done fine-tuning desired imprint borders.',INFO_MESSAGE_DURATION)
                chosen_imprint_desired_mask_bounding_rect = optional_desired_im_rect_coords[best_coords_num[0]]
        if scribble_mode:
            return chosen_imprint_desired_mask_bounding_rect
        else:
            original_rectangle_mask = util.crop_nd_array(desired_im_mask,original_rectangle).astype(np.uint8)
            desired_im_mask = np.zeros_like(desired_im_mask)
            desired_im_mask[chosen_imprint_desired_mask_bounding_rect[1]:chosen_imprint_desired_mask_bounding_rect[1]+chosen_imprint_desired_mask_bounding_rect[3],\
                chosen_imprint_desired_mask_bounding_rect[0]:chosen_imprint_desired_mask_bounding_rect[0]+chosen_imprint_desired_mask_bounding_rect[2]] =\
                util.ResizeCategorialImage(original_rectangle_mask,tuple(chosen_imprint_desired_mask_bounding_rect[2:][::-1]),inclusive=True)
            return chosen_imprint_desired_mask_bounding_rect,desired_im_mask

    def modify_graphical_input_location(self,modification):
        self.SelectImage2Display(self.scribble_display_index)
        self.Undo_scribble(add_2_redo_list=False)
        self.Add_scribble_2_Undo_list()
        if modification in IMPRINT_LOCATION_CHANGES:
            if modification == 'right':
                self.top_left_corner[1] += self.display_zoom_factor
            elif modification == 'left':
                self.top_left_corner[1] -= self.display_zoom_factor
            elif modification == 'up':
                self.top_left_corner[0] -= self.display_zoom_factor
            elif modification == 'down':
                self.top_left_corner[0] += self.display_zoom_factor
        elif modification in IMPRINT_SIZE_CHANGES:
            if modification == 'narrower':
                self.target_imprinting_dimensions[1] -= self.display_zoom_factor
                self.top_left_corner[1] += self.cur_loc_step_size
            elif modification == 'wider':
                self.target_imprinting_dimensions[1] += self.display_zoom_factor
                self.top_left_corner[1] -= self.cur_loc_step_size
            elif modification == 'shorter':
                self.target_imprinting_dimensions[0] -= self.display_zoom_factor
                self.top_left_corner[0] += self.cur_loc_step_size
            elif modification == 'taller':
                self.target_imprinting_dimensions[0] += self.display_zoom_factor
                self.top_left_corner[0] -= self.cur_loc_step_size
            self.cur_loc_step_size = self.display_zoom_factor - self.cur_loc_step_size
        elif modification in IMPRINT_ROTATIONS:
            if self.desired_graphic_input_rotation==0:
                self.non_rotated_desired_graphic_input = 1*self.desired_graphic_input
                self.non_rotated_desired_im_HR_mask_4_imprinting = 1*self.desired_im_HR_mask_4_imprinting
            self.desired_graphic_input_rotation += 1 if 'counter' in modification else -1
            mask_centor_of_mass = np.argwhere(self.non_rotated_desired_im_HR_mask_4_imprinting).mean(0)
            padding_lengths = [np.maximum(0,(self.non_rotated_desired_im_HR_mask_4_imprinting.shape[i]-2*mask_centor_of_mass[i])*np.array([1,-1])).astype(int) for i in range(2)]
            padded_mask = PIL_Image.fromarray(255*np.pad(self.non_rotated_desired_im_HR_mask_4_imprinting,(tuple(padding_lengths[0]),tuple(padding_lengths[1]))))
            padded_input = PIL_Image.fromarray((255*np.pad(self.non_rotated_desired_graphic_input,(tuple(padding_lengths[0]),tuple(padding_lengths[1]),(0,0)))).astype(np.uint8))
            rotated_mask = np.array(padded_mask.rotate(self.desired_graphic_input_rotation)).astype(self.desired_im_HR_mask_4_imprinting.dtype)/255
            rotated_input = np.array(padded_input.rotate(self.desired_graphic_input_rotation)).astype(self.desired_graphic_input.dtype)/255
            self.desired_im_HR_mask_4_imprinting = 1*rotated_mask[padding_lengths[0][0]:util.IndexingHelper(-1*padding_lengths[0][1],True),padding_lengths[1][0]:util.IndexingHelper(-1*padding_lengths[1][1],True)]
            self.desired_graphic_input = 1*rotated_input[padding_lengths[0][0]:util.IndexingHelper(-1*padding_lengths[0][1],True),padding_lengths[1][0]:util.IndexingHelper(-1*padding_lengths[1][1],True),:]
            rotated_mask_indecis = np.argwhere(self.desired_im_HR_mask_4_imprinting).transpose()
            new_bounding_rect = np.array([np.min(rotated_mask_indecis[1]),np.min(rotated_mask_indecis[0])])
            self.imprint_desired_mask_bounding_rect = np.concatenate([new_bounding_rect,np.array([np.max(rotated_mask_indecis[1]), np.max(rotated_mask_indecis[0])])-new_bounding_rect+1])

        self.embed_graphical_input(auto_location=False)

    def finalize_imprinting(self, e=None,transparent_mask=False):
        auto_location_press = self.imprinting_location_boundaries is not None
        if self.last_pos and e is not None:
            # Clear up indicator.
            self.timer_cleanup()
            e_y,e_x,origin_y,origin_x = e.pos().y(),e.pos().x(),self.origin_pos.y(),self.origin_pos.x()
            def convert_2_saved_im_size(coord):
                return int(np.round(coord/self.display_zoom_factor))
            e_y, e_x,origin_y,origin_x = convert_2_saved_im_size(e_y),convert_2_saved_im_size(e_x),convert_2_saved_im_size(origin_y),convert_2_saved_im_size(origin_x)
            self.target_imprinting_dimensions = np.array([np.abs(e_y-origin_y)+1,np.abs(e_x-origin_x)+1])
            # In [y,x] format:
            self.top_left_corner = np.array([np.minimum(e_y,origin_y),np.minimum(e_x,origin_x)])
            # Handling the case of out-of-canvas top left corner:
            # In [height,widht] format:
            self.target_imprinting_dimensions -= np.maximum(0,-1*self.top_left_corner)
            self.top_left_corner = np.maximum(0,self.top_left_corner)
            #     Handling the case of out-of-canvas bottom-right corner:
            self.target_imprinting_dimensions = np.minimum(self.target_imprinting_dimensions,self.HR_size-self.top_left_corner)
        # In [x,y,widht,height] format:
        self.imprint_desired_mask_bounding_rect = np.array(cv2.boundingRect(np.stack([list(p) for p in self.desired_image_HR_mask_vertices], 1).transpose()))
        if self.desired_im_taken_from_same and not auto_location_press and (self.JPEG_GUI or (not (self.last_pos and e is not None) or np.any(self.target_imprinting_dimensions<self.opt['scale']))):
            # If the desired image was taken from the one being edited, this means we want to place the desired image at its original location, to allow subtle region shifting.
            self.top_left_corner = self.imprint_desired_mask_bounding_rect[:2][::-1]
            self.target_imprinting_dimensions = self.imprint_desired_mask_bounding_rect[2:][::-1]-1 #I subtract 1 because the rectangle sizes (coords 2-3) consider both starting and ending coords, which means they add 1
        elif not self.JPEG_GUI and (not (self.last_pos and e is not None) or np.any(self.target_imprinting_dimensions<self.opt['scale'])): #target dimensions are smaller than 2 pixels in the LR image, probably pressed by mistake:
            self.statusBar.showMessage('Attempting to imprint into a too small region. Please try again.',ERR_MESSAGE_DURATION)
            self.reset_mode()
            return
        self.desired_im_HR_mask_4_imprinting = 1 * self.desired_image_HR_mask[0]
        self.desired_graphic_input = 1*self.desired_image[0]
        if transparent_mask:
            GRAYLEVELS_TOLERANCE = 2
            transparency_mask = (np.sqrt(np.mean((np.round(255*self.desired_graphic_input)-self.primary_color.getRgb()[:3])**2,-1))<=GRAYLEVELS_TOLERANCE).astype(np.uint8)
            transparency_mask = cv2.morphologyEx(transparency_mask, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            def mask_negation(mask):
                return (-1 * (mask.astype(np.float32) - 1)).astype(mask.dtype)
            self.desired_im_HR_mask_4_imprinting = np.logical_and(self.desired_im_HR_mask_4_imprinting,mask_negation(transparency_mask)).astype(self.desired_im_HR_mask_4_imprinting.dtype)
        if auto_location_press:
            if self.JPEG_GUI:#Temporary - the auto-location button is used for imprinting into the entire image, in the JPEG configuration:
                pass
                # self.top_left_corner = np.array([0,0])
                # self.target_imprinting_dimensions = np.array(self.HR_size)
            else:
                self.imprinting_location_boundaries.append((origin_y, origin_x, e_y, e_x))
                if len(self.imprinting_location_boundaries) == 1:
                    self.statusBar.showMessage('Registered outer imprinting location bounds',INFO_MESSAGE_DURATION)
                    self.Undo_scribble(add_2_redo_list=False) #Remove the scribble that was just unnecessarily added to the undo list
                    self.reset_mode()
                    return
                else:
                    self.FindOptimalImprintingLocation(self.imprint_desired_mask_bounding_rect)
                    self.statusBar.showMessage('Automatically determined optimal imprinting location',INFO_MESSAGE_DURATION)
                    self.imprinting_location_boundaries = None
        self.embed_graphical_input(auto_location_press)

    def embed_graphical_input(self,auto_location):
        # Extending the target region size to be an integer muliplication of the SR scale factor, otherwise it cannot correspond to any upscale of a low-res region:
        scribble_mode = 'imprint' not in self.mode
        # Extracting scribble mask:
        existing_scribble_mask = qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:,:,0]
        # if not INTEGRATING_SCRIBBLE_IN_MODEL_SIZE:
        model_size_scribble_mask = np.expand_dims(util.ResizeCategorialImage(existing_scribble_mask,
            dsize=tuple([s//self.display_zoom_factor for s in existing_scribble_mask.shape[:2]]),inclusive=True),-1)
        existing_scribble_mask = np.expand_dims(existing_scribble_mask,-1)
        if TRACK_ADDED_SCRIBLES:
            existing_im_2_add_2 = (qimage2ndarray.rgb_view(self.pixmap().toImage())/255).astype(np.float32)
            existing_im_2_add_2 = util.ResizeScribbleImage(existing_im_2_add_2,dsize=tuple([s//self.display_zoom_factor for s in existing_im_2_add_2.shape[:2]]))

        else:
            existing_im_2_add_2 = 1*self.output_image_0_1[0].data.cpu().numpy().transpose(1, 2, 0)


        def padding_params(org_dimensions,org_top_left):
            extended_dimensions = (np.ceil(org_dimensions/self.opt['scale'])*self.opt['scale']).astype(np.uint32)
            if self.JPEG_GUI: # In this case, I need to align the target location with the JPEG blocks grid (the chroma ones). So I want the top left corner to be aligned:
                target_im_pad_sizes = (org_top_left-np.floor(org_top_left/self.opt['scale'])*self.opt['scale']).astype(np.uint32)
                # This pre-padding may lead to negative post-padding values, so I need to extend the extended target dimensions in this case:
                extended_dimensions += (self.opt['scale']*(extended_dimensions-org_dimensions-target_im_pad_sizes<0)).astype(extended_dimensions.dtype)
            else:
                # Calculating pre-padding:
                target_im_pad_sizes = (extended_dimensions - org_dimensions) // 2
                # Transfering padding from post to pre in case image dimensions don't allow a big enough post-padding:
                maximal_possible_post_paddings = np.array(existing_im_2_add_2.shape[:2])-org_dimensions-org_top_left
                target_im_pad_sizes += np.maximum([0,0],extended_dimensions - org_dimensions-target_im_pad_sizes-maximal_possible_post_paddings)
            # Here there is an issue when the target region is very close to canvas borders: As the padding is arbitrary, it may cause the extended location to fall outside the canvas. Didn't take care of it, as it reuqires some thought and it rarely happens.
            actual_top_left_corner = np.maximum([0, 0], org_top_left -target_im_pad_sizes)
            target_im_pad_sizes = np.stack([target_im_pad_sizes,extended_dimensions-org_dimensions-target_im_pad_sizes],-1)
            return extended_dimensions,actual_top_left_corner,target_im_pad_sizes

        def crop_and_rescale_desired_image():
            crop_desired_im_using_bounding_rect = lambda array: util.crop_nd_array(array,self.imprint_desired_mask_bounding_rect)
            cropped_desired_image = crop_desired_im_using_bounding_rect(self.desired_graphic_input)
            cropped_desired_image_mask = crop_desired_im_using_bounding_rect(self.desired_im_HR_mask_4_imprinting)
            rescaled_cropped_desired_image = 255 * util.ResizeScribbleImage(cropped_desired_image, dsize=tuple(self.target_imprinting_dimensions))
            rescaled_cropped_desired_image_mask = np.expand_dims(util.ResizeCategorialImage(cropped_desired_image_mask.astype(np.uint8),
                dsize=tuple(self.target_imprinting_dimensions),inclusive=True), -1)
            return rescaled_cropped_desired_image,rescaled_cropped_desired_image_mask

        def crop_im_and_mask(im,mask):
            def crop_target_im_using_selected_rectangle(array):
                return util.crop_nd_array(array,np.concatenate([actual_top_left_corner[::-1],extended_dimensions[::-1]]))
                # return array[actual_top_left_corner[0]:actual_top_left_corner[0]+extended_dimensions[0],actual_top_left_corner[1]:actual_top_left_corner[1]+extended_dimensions[1],...]
            cropped_im = 255*crop_target_im_using_selected_rectangle(im)
            cropped_mask = crop_target_im_using_selected_rectangle(mask)
            return cropped_im,cropped_mask

        def zero_pad_desired_array(array,padding,mode='constant'):
            return np.pad(array,(tuple(padding[0]),tuple(padding[1]),(0,0)),mode=mode)

        if not scribble_mode:
            extended_dimensions, actual_top_left_corner, target_im_pad_sizes = padding_params(self.target_imprinting_dimensions, self.top_left_corner)
            relevant_existing_scribble_image,relevant_existing_scribble_image_mask = \
                crop_im_and_mask(im=existing_im_2_add_2,mask=model_size_scribble_mask)
        if self.JPEG_GUI and auto_location:  # Imprinting using the "auto-location" button:
            if scribble_mode:
                extended_dims_of_desired, top_left_of_desired, desired_im_pad_sizes = \
                    padding_params(self.imprint_desired_mask_bounding_rect[2:][::-1],self.imprint_desired_mask_bounding_rect[:2][::-1])
                desired_scribble_image = util.crop_nd_array(self.desired_graphic_input,self.imprint_desired_mask_bounding_rect)
                desired_scribble_image = zero_pad_desired_array(desired_scribble_image, desired_im_pad_sizes)
                desired_scribble_image_mask = util.crop_nd_array(self.desired_im_HR_mask_4_imprinting,self.imprint_desired_mask_bounding_rect)
                desired_scribble_image_mask = zero_pad_desired_array(np.expand_dims(desired_scribble_image_mask, -1),desired_im_pad_sizes)
                self.imprint_desired_mask_bounding_rect = np.concatenate([top_left_of_desired[::-1],extended_dims_of_desired[::-1]])
                modified_target_bounding_rect = \
                    self.Find_Optimal_Desired_Imprint_Mask(desired_mask_bounding_rect=np.concatenate([top_left_of_desired[::-1],extended_dims_of_desired[::-1]]),
                        fixed_image=desired_scribble_image, desired_image=255 * self.output_image_0_1[0].data.cpu().numpy().transpose(1, 2, 0),
                        desired_im_mask=self.desired_im_HR_mask_4_imprinting,padding=desired_im_pad_sizes,scribble_mode=True,
                        org_choice_rectangle=np.concatenate([self.top_left_corner[::-1],self.target_imprinting_dimensions[::-1]]))
                self.top_left_corner,self.target_imprinting_dimensions = modified_target_bounding_rect[:2][::-1],modified_target_bounding_rect[2:][::-1]
            else:
                self.imprint_desired_mask_bounding_rect,self.desired_im_HR_mask_4_imprinting = \
                    self.Find_Optimal_Desired_Imprint_Mask(desired_mask_bounding_rect=self.imprint_desired_mask_bounding_rect,
                    fixed_image=relevant_existing_scribble_image,desired_image=255*self.desired_graphic_input,
                       desired_im_mask=self.desired_im_HR_mask_4_imprinting,padding=target_im_pad_sizes)

        IGNORE_DESIRED_MASK_4_COMBINATION = True
        if scribble_mode:
            extended_dimensions, actual_top_left_corner, target_im_pad_sizes = padding_params(self.target_imprinting_dimensions, self.top_left_corner)
            relevant_existing_scribble_image, relevant_existing_scribble_image_mask = \
                crop_im_and_mask(im=existing_im_2_add_2,mask=model_size_scribble_mask)
        rescaled_cropped_desired_image,rescaled_cropped_desired_image_mask = crop_and_rescale_desired_image()
            # Padding desired regions to fit the size of relevant_existing_scribble_image, which is an integer multiplication of scale_facotr*display_factor:
        rescaled_cropped_desired_image = zero_pad_desired_array(rescaled_cropped_desired_image,padding=target_im_pad_sizes,mode='edge' if IGNORE_DESIRED_MASK_4_COMBINATION else 'constant')
        rescaled_cropped_desired_image_mask = zero_pad_desired_array(rescaled_cropped_desired_image_mask,padding=target_im_pad_sizes)
        if self.JPEG_GUI:
            # For JPEG, setting the imprinted image to the existing one everywhere outside the mask. This is because consistency is enforced during imprinting itself, and I only want the desired image where the mask is non-zero, so no point of using it outside the mask too.
            rescaled_cropped_desired_image = rescaled_cropped_desired_image*rescaled_cropped_desired_image_mask+\
                (1-rescaled_cropped_desired_image_mask)*relevant_existing_scribble_image
        if not IGNORE_DESIRED_MASK_4_COMBINATION:
            rescaled_cropped_desired_image = rescaled_cropped_desired_image*rescaled_cropped_desired_image_mask+relevant_existing_scribble_image*(1-rescaled_cropped_desired_image_mask)
        combined_image_2_input = np.clip(255*util.ResizeScribbleImage(self.Enforce_Consistency_on_Image_Pair(relevant_existing_scribble_image/255,
            rescaled_cropped_desired_image/255),dsize=tuple(relevant_existing_scribble_image.shape[:2])),0,255)
        def integrate_patch_into_image(patch,image):
            f = self.display_zoom_factor
            image[f*actual_top_left_corner[0]:f*actual_top_left_corner[0]+f*extended_dimensions[0],f*actual_top_left_corner[1]:f*actual_top_left_corner[1]+f*extended_dimensions[1],...] = patch
            return image
        combined_image_2_input = util.ResizeScribbleImage(combined_image_2_input,dsize=tuple([s*self.display_zoom_factor for s in combined_image_2_input.shape[:2]]))
        new_scribble_image = integrate_patch_into_image(combined_image_2_input,qimage2ndarray.rgb_view(self.pixmap().toImage()))
        self.setPixmap(QPixmap(qimage2ndarray.array2qimage(new_scribble_image)))

        combined_image_mask_2_input = rescaled_cropped_desired_image_mask+(1-rescaled_cropped_desired_image_mask)*relevant_existing_scribble_image_mask
        combined_image_mask_2_input = util.ResizeCategorialImage(combined_image_mask_2_input[:,:,0],dsize=tuple([s*self.display_zoom_factor for s in combined_image_mask_2_input.shape[:2]]),inclusive=True)
        new_scribble_image_mask = integrate_patch_into_image(combined_image_mask_2_input, existing_scribble_mask[:,:,0])
        self.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(np.repeat(np.expand_dims(new_scribble_image_mask,-1),3,-1))))
        self.imprinting_arrows_enabling(True)
        self.reset_mode()

    def imprinting_arrows_enabling(self,enable):
        for button_name in IMPRINT_LOCATION_CHANGES+IMPRINT_SIZE_CHANGES+IMPRINT_ROTATIONS:
            cur_button_enable = bool(enable)
            if enable==True:
                if button_name=='left' or (button_name == 'wider' and self.cur_loc_step_size):
                    cur_button_enable = self.top_left_corner[1] >= self.display_zoom_factor
                elif button_name == 'right' or (button_name == 'wider' and not self.cur_loc_step_size):
                    cur_button_enable = self.top_left_corner[1]+self.target_imprinting_dimensions[1] <= self.HR_size[1]-self.display_zoom_factor
                elif button_name == 'up' or (button_name == 'taller' and self.cur_loc_step_size):
                    cur_button_enable = self.top_left_corner[0] >= self.display_zoom_factor
                elif button_name == 'down' or (button_name == 'taller' and not self.cur_loc_step_size):
                    cur_button_enable = self.top_left_corner[0]+self.target_imprinting_dimensions[0] <= self.HR_size[0]-self.display_zoom_factor
            getattr(self, button_name + '_imprinting_button').setEnabled(cur_button_enable)

    def scribble_undo_redo_enabling(self,enable):
        if enable:
            self.undo_scribble_button.setEnabled(len(self.scribble_history) > 0)
            self.redo_scribble_button.setEnabled(len(self.scribble_redo_list) > 0)
        else:
            for button_name in ['undo','redo']:
                getattr(self, button_name + '_scribble_button').setEnabled(enable)

    # Polygon events

    def polygon_mousePressEvent(self, e):
        self.active_shape_fn = 'drawPolygon'
        self.preview_pen = QPen(PREVIEW_PEN)
        self.generic_poly_mousePressEvent(e)

    def polygon_timerEvent(self, final=False):
        self.generic_poly_timerEvent(final)

    def polygon_mouseMoveEvent(self, e):
        self.generic_poly_mouseMoveEvent(e)

    def polygon_mouseDoubleClickEvent(self, e):
        self.generic_poly_mouseDoubleClickEvent(e)

    # Ellipse events

    def ellipse_mousePressEvent(self, e):
        self.active_shape_fn = 'drawEllipse'
        self.active_shape_args = ()
        self.preview_pen = QPen(PREVIEW_PEN)
        self.generic_shape_mousePressEvent(e)

    def ellipse_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def ellipse_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def ellipse_mouseReleaseEvent(self, e):
        self.generic_shape_mouseReleaseEvent(e)

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
        parser.add_argument('-JPEG', action='store_true', help='JPEG decompresion exploration')
        self.JPEG_GUI =  parser.parse_args().JPEG
        opt_name = None
        self.display_ESRGAN = DISPLAY_ESRGAN_RESULTS
        if self.JPEG_GUI:
            opt_name = 'JPEG_chroma'
            self.display_ESRGAN = False
            self.real_JPEG_image = False

        util.Assign_GPU()
        self.num_random_Zs = NUM_RANDOM_ZS
        self.canvas = Canvas()
        self.setupUi()
        # Replace canvas placeholder from QtDesigner.
        self.horizontalLayout.removeWidget(self.canvas)
        # We need to enable mouse tracking to follow the mouse without the button pressed.
        self.canvas.setMouseTracking(True)
        # Enable focus to capture key inputs.
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.opt = option.parse(parser.parse_args().opt, is_train=False,name=opt_name)
        self.opt = option.dict_to_nonedict(self.opt)
        if DISPLAY_INDUCED_LR:
            #Add a 3rd canvas:
            self.LR_canvas = Canvas()
            self.LR_canvas.initialize()
            self.horizontalLayout.addWidget(self.LR_canvas)
        # Initialize animation timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.on_timer)
        self.timer.setInterval(100)
        self.timer.start()

        # Initializing some variables:
        self.auto_set_hist_temperature = False
        self.latest_optimizer_objective = ''
        self.canvas.display_zoom_factor = 1*DISPLAY_ZOOM_FACTOR
        self.canvas.desired_image = None
        self.canvas.imprinting_location_boundaries = None
        self.using_estimated_kernel = False
        self.canvas.within_drawing = False # I add this to distinguish between first mouse press (initiating polygon drawing) and the rest of the presses. The motivation is to know when I BEGIN a scribble action.
        self.canvas.set_primary_color('#000000')
        self.canvas.color_state = 0
        self.canvas.cyclic_color_shift = 0
        self.canvas.local_TV_identifier  = LOCAL_TV_MASK_IDENTIFIERS_RANGE[0]-1
        self.canvas.in_picking_desired_hist_mode = False
        self.canvas.current_display_index = 1*self.cur_Z_im_index

        # More initializations:
        self.canvas.Z_optimizer_Reset()
        self.canvas.opt = self.opt
        self.canvas.initialize()
        self.canvas.HR_Z = False if self.JPEG_GUI else ('HR' in self.canvas.opt['network_G']['latent_input_domain'])
        self.canvas.H_L_domains_ratio = 8 if self.JPEG_GUI else self.opt['scale']
        self.Initialize_SR_model(reprocess=False)
        self.canvas.latest_scribble_color_reset = time.time()

        # Assigning handles of some buttons and functions to canvas:
        button_handles_required_by_canvas = ['undo_scribble','redo_scribble','special_behavior','FoolAdversary','selectrect',
            'selectpoly','IncreasePeriodicity_1D','IncreasePeriodicity_2D','indicatePeriodicity','periodicity_mag_1','periodicity_mag_2',
            'apply_scribble','loop_apply_scribble','DisplayedImageSelection']+[b+'_imprinting' for b in (IMPRINT_SIZE_CHANGES+IMPRINT_LOCATION_CHANGES+IMPRINT_ROTATIONS)]
        for button in button_handles_required_by_canvas:
            setattr(self.canvas,button+'_button',getattr(self,button+'_button'))
        self.canvas.scribble_modes = SCRIBBLE_MODES
        self.canvas.Update_Image_Display = self.Update_Image_Display
        self.canvas.SelectImage2Display = self.SelectImage2Display
        if self.JPEG_GUI:
            self.estimatedKenrel_button.setEnabled(False)
            self.FoolAdversary_button.setEnabled(False)
            self.open_image_button.setEnabled(False) #Loading existing jpg files is not yet enabled
        self.canvas.Enforce_Consistency_on_Image_Pair = self.canvas.SR_model.Enforce_pair_Consistency if self.JPEG_GUI else self.canvas.SR_model.CEM_net.Enforce_DT_on_Image_Pair
        self.canvas.Project_2_Orthog_Nullspace = None if self.JPEG_GUI else self.canvas.SR_model.CEM_net.Project_2_ortho_2_NS
        self.canvas.statusBar = self.statusBar
        self.canvas.JPEG_GUI = self.JPEG_GUI

        #### Connecting buttons to functions:
        # Load & Save:
        self.open_image_button.clicked.connect(lambda x: self.open_file(HR_image=False))
        self.open_HR_image_button.clicked.connect(lambda x: self.open_file(HR_image=True))
        self.Z_load_button.pressed.connect(self.Load_Z)
        self.SaveImageAndData_button.clicked.connect(self.save_file_and_Z_map)
        if self.JPEG_GUI:
            self.QF_box.valueChanged.connect(lambda s: self.Update_QF(s))
        # Display:
        self.Zdisplay_button.clicked.connect(self.ToggleDisplay_Z_Image)
        self.DecreaseDisplayZoom_button.clicked.connect(lambda x: self.Change_Display_Zoom(increase=False))
        self.Safely_Enable_Zoom_Buttons(True)
        self.IncreaseDisplayZoom_button.clicked.connect(lambda x: self.Change_Display_Zoom(increase=True))
        self.CopyFromAlternative_button.pressed.connect(self.CopyAlternative2Default)
        self.Copy2Alternative_button.pressed.connect(self.CopyDefault2Alternative)
        # Region selection:
        self.unselect_button.clicked.connect(self.Clear_Z_Mask)
        self.invertSelection_button.clicked.connect(self.Invert_Z_Mask)
        self.Z_mask_load_button.clicked.connect(self.Load_Z_mask)
        # Reference & General:
        self.desiredExternalAppearanceMode_button.clicked.connect(lambda checked: self.DesiredAppearanceMode(checked,another_image=True))
        self.desiredAppearanceMode_button.clicked.connect(lambda checked: self.DesiredAppearanceMode(checked,another_image=False))
        self.undoZ_button.clicked.connect(self.Undo_Z)
        self.redoZ_button.clicked.connect(self.Redo_Z)
        self.estimatedKenrel_button.pressed.connect(self.Change_kernel_in_use)
        # Scribbling:
        self.sizeselect_slider.valueChanged.connect(lambda s: self.canvas.set_config('size', s))
        for mode in MODES:
            btn = getattr(self, '%s_button' % mode)
            btn.pressed.connect(lambda mode=mode: self.canvas.set_mode(mode))
        # Manage scrible:
        self.scribble_reset_button.pressed.connect(self.Reset_Image_4_Scribbling)
        self.apply_scribble_button.clicked.connect(lambda x:self.Optimize_Z('scribble'))
        self.loop_apply_scribble_button.clicked.connect(lambda x:self.Optimize_Z('scribble',loop=True))
        self.canvas.color_button.pressed.connect(lambda: self.choose_color(self.canvas.set_primary_color))
        self.canvas.cycleColorState_button.pressed.connect(self.canvas.cycle_color_state)
        self.canvas.undo_scribble_button.clicked.connect(lambda s:self.canvas.Undo_scribble(add_2_redo_list=True))
        self.canvas.redo_scribble_button.clicked.connect(self.canvas.Redo_scribble)
        # Imprinting:
        self.up_imprinting_button.clicked.connect(lambda s:self.canvas.modify_graphical_input_location(modification='up'))
        self.down_imprinting_button.clicked.connect(lambda s:self.canvas.modify_graphical_input_location(modification='down'))
        self.right_imprinting_button.clicked.connect(lambda s:self.canvas.modify_graphical_input_location(modification='right'))
        self.left_imprinting_button.clicked.connect(lambda s:self.canvas.modify_graphical_input_location(modification='left'))
        self.wider_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='wider'))
        self.taller_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='taller'))
        self.narrower_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='narrower'))
        self.shorter_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='shorter'))
        self.clockwise_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='clockwise'))
        self.counter_clockwise_imprinting_button.clicked.connect(lambda s: self.canvas.modify_graphical_input_location(modification='counter_clockwise'))
        # Optimize Z:
        self.IncreaseSTD_button.clicked.connect(lambda x:self.Optimize_Z('STD_increase' if RELATIVE_STD_OPT else 'max_STD', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.DecreaseSTD_button.clicked.connect(lambda x:self.Optimize_Z('STD_decrease' if RELATIVE_STD_OPT else 'min_STD', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.DecreaseTV_button.clicked.connect(lambda x:self.Optimize_Z('TV', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.ImitateHist_button.clicked.connect(lambda x:self.Optimize_Z('hist', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.ImitatePatchHist_button.clicked.connect(lambda x:self.Optimize_Z('patchhist', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.FoolAdversary_button.clicked.connect(lambda x:self.Optimize_Z('Adversarial', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.STD_increment.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.ProcessRandZ_button.clicked.connect(lambda x: self.Process_Random_Z(limited=False))
        self.ProcessLimitedRandZ_button.clicked.connect(lambda x: self.Process_Random_Z(limited=True))
        if self.auto_hist_temperature_mode_Enabled:
            self.auto_hist_temperature_mode_button.clicked.connect(lambda checked:self.AutoHistTemperatureMode(checked))
        # Periodicity:
        self.IncreasePeriodicity_2D_button.clicked.connect(lambda x:self.Optimize_Z('periodicity', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.IncreasePeriodicity_1D_button.clicked.connect(lambda x:self.Optimize_Z('periodicity_1D', loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS))
        self.canvas.periodicity_mag_1_button.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.canvas.periodicity_mag_2_button.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        # Uniform Z control:
        self.canvas.Z0_slider.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=0,dont_update_undo_list=True))
        self.canvas.Z0_slider.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.Z0_slider.value() / 100, index=0))
        self.canvas.Z1_slider.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=1,dont_update_undo_list=True))
        self.canvas.Z1_slider.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.Z1_slider.value() / 100, index=1))
        self.canvas.third_channel_slider.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=2,dont_update_undo_list=True))
        self.canvas.third_channel_slider.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.third_channel_slider.value() / 100, index=2))
        self.uniformZ_button.clicked.connect(self.ApplyUniformZ)

        if self.JPEG_GUI:
            #     HSV manipulation:
            self.H_clockwise_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='H',increase=True))
            self.H_counter_clockwise_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='H',increase=False))
            self.S_up_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='S',increase=True))
            self.S_down_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='S',increase=False))
            self.V_up_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='V',increase=True))
            self.V_down_button.clicked.connect(lambda s: self.Manipulate_HSV(channel='V',increase=False))

        # Scribble mask:
        self.canvas.scribble_mask_canvas = Canvas()
        self.canvas.scribble_mask_canvas.initialize()

        # Setting GUI's location and geometry:
        self.setGeometry(QRect(0,0,self.sizeHint().width(),self.sizeHint().height()))

        # Loading and downsampling an initial demo HR image:
        sample_image_folder = '../Samples'
        example_image = [f for f in os.listdir(sample_image_folder) if any([ext in f for ext in ['.png','.jpg','.bmp']])][0]
        self.open_file(path=os.path.join(sample_image_folder,example_image),HR_image=True,canvas_pos=(self.geometry().getCoords()[2]+HORIZ_MAINWINDOW_OFFSET,self.geometry().getCoords()[1]))

        self.show()

    def Manipulate_HSV(self,channel,increase):
        STEP_SIZE = 0.05 # 0.01
        YCbCrnotHSV = False
        self.canvas.scribble_mode_entry_operations()
        existing_scribble_im = qimage2ndarray.rgb_view(self.canvas.pixmap().toImage())
        if YCbCrnotHSV:
            HSV_image = data_util.bgr2ycbcr(existing_scribble_im[:,:,[2,1,0]]/255,only_y=False)[:,:,[2,1,0]]
        else:
            HSV_image = rgb2hsv(existing_scribble_im)
        multiplier = 1.01 if increase else 0.99
        adder = STEP_SIZE if increase else -1*STEP_SIZE
        if channel=='H':
            if YCbCrnotHSV:
                HSV_image[:, :, 0] = np.clip(HSV_image[:, :, 0] + self.canvas.HR_mask_display_size * adder, 0,1)
            else:
                HSV_image[:,:,0] = np.mod(HSV_image[:,:,0]+self.canvas.HR_mask_display_size*adder,1)
        elif channel=='S':
            HSV_image[:,:,1] = np.clip(HSV_image[:,:,1]+self.canvas.HR_mask_display_size*adder,0,1)
        elif channel=='V':
            HSV_image[:,:,2] = np.clip(HSV_image[:,:,2]+self.canvas.HR_mask_display_size*adder,0,1)
        if YCbCrnotHSV:
            RGB_image = util.ResizeScribbleImage(data_util.ycbcr2rgb(HSV_image[:,:,[2,1,0]]).astype(np.float32),dsize=tuple(self.canvas.HR_size))
        else:
            RGB_image = util.ResizeScribbleImage(hsv2rgb(HSV_image).astype(np.float32),dsize=tuple(self.canvas.HR_size))
        RGB_image = self.canvas.Enforce_Consistency_on_Image_Pair(self.canvas.output_image_0_1[0].data.cpu().numpy().transpose(1, 2, 0),RGB_image)
        RGB_image = 255*util.ResizeScribbleImage(RGB_image,dsize=self.canvas.HR_mask_display_size.shape)
        # Changing only the parts of scribble image & mask that fall within the (Jpeg-smeared) currently selected area:
        blocks_smeared_mask = np.expand_dims(self.canvas.HR_mask_smeared_2_blocks_display_size,-1)
        RGB_image = blocks_smeared_mask*RGB_image+(1-blocks_smeared_mask)*existing_scribble_im
        self.canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(RGB_image.astype(np.uint8))))
        existing_scribble_mask = qimage2ndarray.rgb_view(self.canvas.scribble_mask_canvas.pixmap().toImage())
        new_scribble_mask = blocks_smeared_mask*np.repeat(np.expand_dims(self.canvas.HR_mask_display_size, -1), 3, -1)+(1-blocks_smeared_mask)*existing_scribble_mask
        self.canvas.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(new_scribble_mask)))
        self.canvas.scribble_mode_exit_operations()

    def Change_kernel_in_use(self,use_estimated=None):
        if use_estimated is None:
            self.using_estimated_kernel = not self.estimatedKenrel_button.isChecked()
        else:
            self.using_estimated_kernel = use_estimated
            self.estimatedKenrel_button.setChecked(use_estimated)
        if self.using_estimated_kernel:
            if self.estimated_kernel is None: #estimate the SR kernel, using the KernelGAN method
                self.statusBar.showMessage('Estimating SR kernel using the KernelGAN method. Please hold on...')
                # print()
                KernelGAN_conf = KernelGAN.Config().parse(['--X4'] if self.canvas.opt['scale']==4 else [])
                KernelGAN_conf.LR_image = self.LR_image
                self.estimated_kernel = KernelGAN.train(KernelGAN_conf)
                self.statusBar.showMessage('Kernel estimation by the KernelGAN method is done.',INFO_MESSAGE_DURATION)
            kernel_4_model = self.estimated_kernel
        else:
            kernel_4_model = 'reset_2_default'
        self.Initialize_SR_model(kernel=kernel_4_model)
        # Disabling ESRGAN result display to prevent misleading - ESRGAN cannot use the estimated kernel: (This is primarily for the (frequent) cases of the estimated kernel inducing poor results.)
        self.DisplayedImageSelection_button.model().item(self.ESRGAN_index).setEnabled(not self.using_estimated_kernel)

    def choose_color(self, callback):
        dlg = QColorDialog(self.canvas.primary_color)
        if dlg.exec():
            callback( dlg.selectedColor().name() )

    def AutoHistTemperatureMode(self,checked):
        if checked:
            self.auto_set_hist_temperature = True
            self.canvas.Z_optimizer_Reset()
        else:
            self.auto_set_hist_temperature = False

    def DesiredAppearanceMode(self,checked,another_image):
        self.canvas.in_picking_desired_hist_mode = 1*checked
        self.canvas.using_external_image = 1*another_image
        if checked:
            self.Safely_Enable_Zoom_Buttons(False)
            self.MasksStorage(True)
            self.canvas.selection_display_timer_cleanup(clear_data=True)
            self.canvas.desired_im_taken_from_same = not another_image
            if self.canvas.current_display_index == self.canvas.scribble_display_index:
                self.Update_Scribble_Data()
            if another_image:
                path, _ = QFileDialog.getOpenFileName(self, "Desired image for histogram imitation", "",IMAGE_FILE_EXST_FILTER)
                if path:
                    self.canvas.desired_image = data_util.read_img(None, path)
                    if self.canvas.desired_image.ndim > 2 and self.canvas.desired_image.shape[2] == 3:
                        self.canvas.desired_image = self.canvas.desired_image[:, :, [2, 1, 0]]
                    else:
                        self.canvas.desired_image = np.repeat(self.canvas.desired_image.reshape([self.canvas.desired_image.shape[0],self.canvas.desired_image.shape[1],1]), 3, 2)
                    if self.JPEG_GUI:
                        self.canvas.desired_image = self.canvas.desired_image[:self.canvas.desired_image.shape[0]//16*16,
                                                    :self.canvas.desired_image.shape[1]//16*16,...]
                    current_geometry = self.canvas.geometry().getCoords()
                    self.canvas.setGeometry(QRect(current_geometry[0], current_geometry[1],self.canvas.desired_image.shape[1],self.canvas.desired_image.shape[0]))
                    self.canvas.setWindowTitle('Importing from %s'%(path.split('/')[-1]))
                    # self.canvas.show()
                    im_2_display = 1*self.canvas.desired_image
                    pixmap = QPixmap()
                    pixmap.convertFromImage(qimage2ndarray.array2qimage(255*im_2_display))
                    self.canvas.setPixmap(pixmap)
                    # Removed the following line because I now rescale the desired image to canvas size anyway...
                    # self.canvas.setGeometry(QRect(0,0,self.canvas.desired_image.shape[0],self.canvas.desired_image.shape[1]))
                    self.canvas.HR_size = list(self.canvas.desired_image.shape[:2])
            else:
                self.canvas.desired_image = self.canvas.output_image_0_1[0].data.cpu().numpy().transpose(1,2,0)
            self.canvas.HR_selected_mask = np.ones(self.canvas.desired_image.shape[:2])
            self.canvas.LR_mask_vertices = [(0, 0), tuple(self.canvas.LR_size[::-1])]
            self.canvas.HR_mask_vertices = [(0, 0), tuple(self.canvas.HR_size[::-1])]
        else:
            self.Safely_Enable_Zoom_Buttons(True)
            self.canvas.desired_graphic_input_rotation = 0
            self.canvas.desired_image_HR_mask = 1*self.canvas.HR_selected_mask
            self.canvas.desired_image_HR_mask_vertices = self.canvas.HR_mask_vertices
            self.MasksStorage(False)
            self.Update_Image_Display()
            self.ImitateHist_button.setEnabled(True)
            self.ImitatePatchHist_button.setEnabled(True)
            self.imprinting_auto_location_button.setEnabled(True)
            self.imprinting_button.setEnabled(True)
            self.canvas.desired_image,self.canvas.desired_image_HR_mask = [self.canvas.desired_image],[self.canvas.desired_image_HR_mask] #Warpping in a list to have a unified framework for the case of transformed hist image versions.
            if another_image:
                self.Update_Canvas_Size_and_Title()
            if DOWNSCALED_HIST_VERSIONS:
                raise Exception('I removed the per-scale temperature modification in the histogram forward function, since I could not tell why it was there.')
                cur_downscaling_factor = DOWNSCALED_HIST_VERSIONS
                while cur_downscaling_factor>MIN_DOWNSCALING_4_HIST:
                    self.canvas.desired_image.append(cv2.resize(self.canvas.desired_image[0],dsize=None,fx=cur_downscaling_factor,fy=cur_downscaling_factor))
                    self.canvas.desired_image_HR_mask.append((cv2.resize(self.canvas.desired_image_HR_mask[0],dsize=None,fx=cur_downscaling_factor,
                        fy=cur_downscaling_factor)>0.5).astype(self.canvas.desired_image_HR_mask[0].dtype))
                    cur_downscaling_factor *= DOWNSCALED_HIST_VERSIONS

    def Compute_SR_Image(self,dont_update_undo_list=False):
        if self.cur_Z.size(2)==1:
            cur_Z = ((self.cur_Z * torch.ones([1, 1] + self.canvas.Z_size) - 0.5) * 2).type(self.var_L.type())
        else:
            cur_Z = self.cur_Z.type(self.var_L.type())
        self.Feed_n_Run_model(cur_Z)
        if DISPLAY_INDUCED_LR:
            self.induced_LR_image = self.canvas.SR_model.netG.module.DownscaleOP(self.canvas.output_image_0_1)

    def Feed_n_Run_model(self,cur_Z):
        DEBUG_CHROMA = False
        self.canvas.SR_model.Prepare_Input(self.var_L, latent_input=torch.zeros_like(cur_Z) if DEBUG_CHROMA else cur_Z,compressed_input=True)
        if DEBUG_CHROMA:
            print('Y Z STD:%.4f'%(self.canvas.SR_model.GetLatent().std()))
        if self.JPEG_GUI:
            if DEBUG_CHROMA:
                self.canvas.SR_model.test(chroma_input=self.chroma_input,chroma_Z=cur_Z)
                print('chroma Z STD:%.4f' % (self.canvas.SR_model.GetLatent().std()))
            else:
                self.canvas.SR_model.test(chroma_input=self.chroma_input)
        else:
            self.canvas.SR_model.test()
        self.canvas.output_image_0_1 = self.canvas.SR_model.Output_Batch(within_0_1=True)

    def DrawRandChannel(self,min_val,max_val,uniform=False):
        return (max_val-min_val)*torch.rand([1,1]+([1,1] if uniform else self.canvas.Z_size))+min_val

    def Reset_Image_4_Scribbling(self):
        # Delete any scribbled info in selected region, or initialize scribble canvas and mask
        if self.canvas.image_4_scribbling is None:
            self.canvas.image_4_scribbling = 255*self.canvas.random_Z_images[0].detach().float().cpu().numpy().transpose(1, 2, 0).copy()
            self.canvas.current_scribble_mask = np.zeros(self.canvas.HR_size).astype(np.uint8)
            self.Update_Scribble_Mask_Canvas(initialize=True)
            self.Initialize_Image_4_Scribbling_Display_Size()
        else:
            self.canvas.Add_scribble_2_Undo_list()
            if self.canvas.current_display_index == self.canvas.scribble_display_index:#If we are in scribble mode (display), and I want to reset only the masked part,
                # I want to make sure the rest is saved before using the saved part for the non-masked region.
                self.Update_Scribble_Data()
            self.canvas.image_4_scribbling = np.expand_dims(1-self.canvas.HR_selected_mask,-1)*self.canvas.image_4_scribbling +\
                np.expand_dims(self.canvas.HR_selected_mask,-1)*255 * self.canvas.random_Z_images[0].detach().float().cpu().numpy().transpose(1, 2, 0).copy()
            self.canvas.current_scribble_mask = (1-self.canvas.HR_selected_mask).astype(np.bool)*self.canvas.current_scribble_mask+\
                self.canvas.HR_selected_mask.astype(np.bool)*np.zeros(self.canvas.HR_size).astype(np.bool)
            self.Update_Scribble_Mask_Canvas(initialize=False)
            if self.canvas.current_display_index == self.canvas.scribble_display_index:#In case we currently display scribble, momentarily changing display for the scribble erasing to be visible
                self.SelectImage2Display(self.cur_Z_im_index)
                self.SelectImage2Display(self.canvas.scribble_display_index)

        self.apply_scribble_button.setEnabled(False)
        self.loop_apply_scribble_button.setEnabled(False)

    def Initialize_Image_4_Scribbling_Display_Size(self):
        self.canvas.image_4_scribbling_display_size = 1*self.canvas.image_4_scribbling
        if self.canvas.display_zoom_factor>1:
            self.canvas.image_4_scribbling_display_size = util.ResizeScribbleImage(self.canvas.image_4_scribbling_display_size,
                                     tuple([self.canvas.display_zoom_factor * val for val in self.canvas.HR_size]))

    def Update_Scribble_Mask_Canvas(self,initialize=False):
        # Apply saved scribble mask into scribble mask canvas itself. When inititalize, ignore selected region and apply to entire image.
        pixmap = QPixmap()
        if self.canvas.display_zoom_factor>1:
            updating_image = util.ResizeCategorialImage(self.canvas.current_scribble_mask,dsize=tuple([self.canvas.display_zoom_factor*val for val in self.canvas.HR_size]))
        else:
            updating_image = self.canvas.current_scribble_mask
        if initialize:
            pixmap_image = qimage2ndarray.array2qimage(updating_image)
        else:
            pixmap_image = qimage2ndarray.array2qimage(self.canvas.HR_mask_display_size*updating_image+(1-self.canvas.HR_mask_display_size)*qimage2ndarray.rgb_view(self.canvas.scribble_mask_canvas.pixmap().toImage()).mean(2))
        pixmap.convertFromImage(pixmap_image)
        self.canvas.scribble_mask_canvas.setPixmap(pixmap)

    def Reset_Scribbling_Image_Background(self):
        # Update saved scribble with current model's output, wherever not scribbled.
        current_image_display_size = util.ResizeScribbleImage(self.canvas.random_Z_images[0].data.cpu().numpy().transpose((1,2,0)),
            tuple([self.canvas.display_zoom_factor*val for val in self.canvas.HR_size]))
        scribbled_mask = qimage2ndarray.rgb_view(self.canvas.scribble_mask_canvas.pixmap().toImage()).mean(2)>0
        self.canvas.image_4_scribbling_display_size = np.expand_dims(scribbled_mask>0,-1)*self.canvas.image_4_scribbling_display_size+ \
            np.clip(255*np.expand_dims(scribbled_mask==0,-1)*current_image_display_size,0,255).astype(self.canvas.image_4_scribbling_display_size.dtype)

    def Update_Scribble_Data(self):
        # Save graphical scribbles data (e.g. when going OUT of scribble mode)
        Z_displayed = 1*self.Zdisplay_button.isChecked()
        if Z_displayed:
            self.Zdisplay_button.setChecked(False)
            self.Update_Image_Display()
        self.canvas.image_4_scribbling = qimage2ndarray.rgb_view(self.canvas.pixmap().toImage())
        self.canvas.image_4_scribbling_display_size = 1*self.canvas.image_4_scribbling
        self.canvas.current_scribble_mask = qimage2ndarray.rgb_view(self.canvas.scribble_mask_canvas.pixmap().toImage())[:, :, 0]
        if self.canvas.display_zoom_factor>1:
            self.canvas.image_4_scribbling = util.ResizeScribbleImage(self.canvas.image_4_scribbling,dsize=tuple(self.canvas.HR_size))
            self.canvas.current_scribble_mask = util.ResizeCategorialImage(image=self.canvas.current_scribble_mask,dsize=tuple(self.canvas.HR_size))
        if Z_displayed:
            self.Zdisplay_button.setChecked(True)
            self.Update_Image_Display()

    def SelectImage2Display(self,chosen_index=None):
        no_Z_displays = [self.canvas.scribble_display_index,self.GT_HR_index,self.ESRGAN_index,self.input_display_index]
        if chosen_index is not None and chosen_index!=self.canvas.current_display_index:
            self.DisplayedImageSelection_button.setCurrentIndex(chosen_index)# For the case when called not by the DisplayedImageSelectionButton interface
            if self.canvas.current_display_index == self.canvas.scribble_display_index:
                self.Update_Scribble_Data()
            self.canvas.current_display_index = chosen_index
            if self.canvas.current_display_index in no_Z_displays:
                self.Zdisplay_button.setChecked(False)
            self.Zdisplay_button.setEnabled(self.canvas.current_display_index not in no_Z_displays)
            if chosen_index==self.canvas.scribble_display_index:
                self.Reset_Scribbling_Image_Background()
                self.Update_Image_Display()
        self.CopyFromAlternative_button.setEnabled(self.canvas.current_display_index in self.random_display_indexes)
        if self.canvas.current_display_index in [self.cur_Z_im_index,self.canvas.scribble_display_index]:
            # self.canvas.SR_model.fake_H = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
            self.canvas.output_image_0_1 = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
            self.Z_2_display = self.cur_Z
        elif self.canvas.current_display_index in self.random_display_indexes:
            # self.canvas.SR_model.fake_H = 1*self.canvas.random_Z_images[self.canvas.current_display_index-self.random_display_indexes[0]+1].unsqueeze(0)
            self.canvas.output_image_0_1 = 1*self.canvas.random_Z_images[self.canvas.current_display_index-self.random_display_indexes[0]+1].unsqueeze(0)
            self.Z_2_display = self.canvas.random_Zs[self.canvas.current_display_index-self.random_display_indexes[0],...].unsqueeze(0)
        else:
            self.Z_2_display = self.no_Z_image
            if self.canvas.current_display_index==self.GT_HR_index:
                # self.canvas.SR_model.fake_H = 1*self.GT_HR
                self.canvas.output_image_0_1 = 1*self.GT_HR
            elif self.canvas.current_display_index==self.ESRGAN_index:
                # self.canvas.SR_model.fake_H = 1*self.ESRGAN_SR
                self.canvas.output_image_0_1 = 1*self.ESRGAN_SR
            elif self.canvas.current_display_index==self.canvas.scribble_display_index:
                pass#I think there is nothing to do here, because I don't assign the image to SR_model.fake_H
            elif self.canvas.current_display_index==self.input_display_index:
                self.canvas.output_image_0_1 = 1*self.input_image
        if not self.canvas.current_display_index==self.canvas.scribble_display_index:
            self.Update_Image_Display()
        self.canvas.scribble_undo_redo_enabling(enable=(self.canvas.current_display_index==self.canvas.scribble_display_index))
        self.Z_undo_redo_enabling(enable=(self.canvas.current_display_index==self.cur_Z_im_index))

    def CopyAlternative2Default(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        self.cur_Z = (self.canvas.random_Zs[self.canvas.current_display_index-self.random_display_indexes[0],...].to(self.cur_Z.device)*Z_mask+self.cur_Z[0]*(1-Z_mask)).unsqueeze(0)
        self.ReProcess(chosen_display_index=self.cur_Z_im_index)
        self.canvas.Z_optimizer_Reset()
        self.DeriveControlValues()

    def CopyDefault2Alternative(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        for random_Z_num in range(len(self.canvas.random_Zs)):
            self.canvas.random_Zs[random_Z_num] = self.canvas.random_Zs[random_Z_num]*(1-Z_mask)+self.cur_Z[0]*Z_mask
        self.Process_Z_Alternatives()

    def Process_Z_Alternatives(self):
        stored_Z = 1*self.cur_Z
        for i, random_Z in enumerate(self.canvas.random_Zs):
            self.cur_Z = 1*random_Z.unsqueeze(0)
            self.Compute_SR_Image()
            self.canvas.random_Z_images[i + 1] = self.canvas.output_image_0_1[0]
        self.cur_Z = 1*stored_Z
        self.DisplayedImageSelection_button.setEnabled(True)
        self.SelectImage2Display()

    def Process_Random_Z(self,limited):
        if self.num_random_Zs>1 or limited:
            self.Optimize_Z('random_' + ('VGG' if VGG_RANDOM_DOMAIN else 'l1') + ('_limited' if limited else ''), loop=LOOP_IN_ALL_Z_OPTIMIZATION_TOOLS)
        else:
            UNIFORM_RANDOM = False
            Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
            self.canvas.control_values = Z_mask*torch.stack([self.DrawRandChannel(0,self.max_SVD_Lambda,uniform=UNIFORM_RANDOM),
                self.DrawRandChannel(0,self.max_SVD_Lambda,uniform=UNIFORM_RANDOM),self.DrawRandChannel(0,np.pi,uniform=UNIFORM_RANDOM)],
                0).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.control_values
            self.Recompose_cur_Z()
            self.canvas.Update_Z_Sliders()
            self.ReProcess()

    def Validate_Z_optimizer(self,objective):
        if self.latest_optimizer_objective!=objective:# or objective=='hist': # Resetting optimizer in the 'patchhist' case because I use automatic tempersture search there, so I want to search each time for the best temperature.
            self.canvas.Z_optimizer_Reset()

    def MasksStorage(self,store):
        canvas_keys = ['Z_mask','HR_mask_display_size','HR_selected_mask','LR_mask_vertices','HR_size','random_Zs','image_4_scribbling','current_scribble_mask',
                       'selection_display','existing_selection_timer_event','HR_mask_vertices']
        self_keys = ['cur_Z','var_L']+(['chroma_input'] if self.JPEG_GUI else [])
        for key in canvas_keys:
            if store:
                setattr(self,'stored_canvas_%s'%(key),getattr(self.canvas,key))
            else:
                setattr(self.canvas,key, getattr(self, 'stored_canvas_%s' % (key)))
        for key in self_keys:
            if store:
                setattr(self,'stored_%s'%(key),getattr(self,key))
            else:
                setattr(self,key, getattr(self, 'stored_%s' % (key)))

    def SVD_ValuesStorage(self,store):
        if store:
            self.stored_control_values = 1*self.canvas.control_values
        else:
            self.canvas.control_values = 1*self.stored_control_values

    def Crop2BoundingRect(self,arrays,bounding_rect,SF=1):
        operating_on_list = isinstance(arrays,list)
        if not operating_on_list:
            arrays = [arrays]
        if SF!=self.canvas.H_L_domains_ratio:
            bounding_rect = self.canvas.H_L_domains_ratio/SF * bounding_rect
            assert np.all(np.round(bounding_rect)==bounding_rect)
            bounding_rect = bounding_rect.astype(np.int32)
        bounding_rect = 1 * bounding_rect
        arrays_2_return = []
        for array in arrays:
            if isinstance(array,np.ndarray):
                arrays_2_return.append(array[bounding_rect[1]:bounding_rect[1]+bounding_rect[3],bounding_rect[0]:bounding_rect[0]+bounding_rect[2]])
            elif torch.is_tensor(array):
                if array.dim()==4:
                    arrays_2_return.append(array[:,:,bounding_rect[1]:bounding_rect[1] + bounding_rect[3],bounding_rect[0]:bounding_rect[0] + bounding_rect[2]])
                elif array.dim()==2:
                    arrays_2_return.append(array[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],bounding_rect[0]:bounding_rect[0] + bounding_rect[2]])
                else:
                    raise Exception('Unsupported')
        return arrays_2_return if operating_on_list else arrays_2_return[0]

    def Set_Extreme_SVD_Values(self,min_not_max):
        raise Exception('No longer supported - to re-enable, adjust to all recent changes')
        self.SetZ(0 if min_not_max else 1,0,reset_optimizer=False) # I'm using 1 as maximal value and not MAX_LAMBDA_VAL because I want these images to correspond to Z=[-1,1] like in the model training. Different maximal Lambda values will be manifested in the Z optimization itself when cur_Z is normalized.
        self.SetZ(0 if min_not_max else 1, 1,reset_optimizer=False)

    def Crop_masks_2_BoundingRect(self,bounding_rect):
        HR_keys = ['canvas.HR_selected_mask','canvas.SR_model.fake_H','canvas.image_4_scribbling','canvas.current_scribble_mask',
                   'canvas.output_image_0_1','canvas.SR_model.output_image']+(['canvas.Z_mask','cur_Z'] if self.canvas.HR_Z else [])
        LR_keys = ['var_L']+(['canvas.Z_mask','cur_Z'] if not self.canvas.HR_Z else [])
        keys = HR_keys+LR_keys+(['chroma_input'] if self.JPEG_GUI else [])
        def Return_Inner_Attr(key):
            keys = key.split('.')
            fetched_attr = self
            for cur_key in keys:
                fetched_attr = getattr(fetched_attr,cur_key)
            return fetched_attr
        def Set_Inner_Attr(key,value):
            keys = key.split('.')
            fetched_attr = self
            for cur_key in keys[:-1]:
                fetched_attr = getattr(fetched_attr,cur_key)
            setattr(fetched_attr,keys[-1],value)

        for key in keys:
            cur_attr = Return_Inner_Attr(key)
            if isinstance(cur_attr,np.ndarray):
                cur_attr_size = list(cur_attr.shape[:2])
            else:
                cur_attr_size = list(cur_attr.size()[2:])
            size_factor = [self.canvas.HR_size[i]/cur_attr_size[i] for i in range(2)]
            assert size_factor[0]==size_factor[1],'Unisotropic scale factor'
            Set_Inner_Attr(key, self.Crop2BoundingRect(cur_attr, bounding_rect=bounding_rect, SF=size_factor[0]))

    def Optimize_Z(self,objective,loop=False):
        if self.special_behavior_button.isChecked():
            objective = objective.replace('STD','local_Mag').replace('periodicity','periodicityPlus')
        if LOCAL_STD_4_OPT:
            objective = objective.replace('STD','local_STD').replace('periodicity','local_STD_periodicity').replace('TV','local_STD_TV')
        elif L1_REPLACES_HISTOGRAM:
            objective = objective.replace('hist', 'l12GT')
        if NO_DC_IN_PATCH_HISTOGRAM:
            objective = objective.replace('hist','hist_noDC_no_localSTD' if self.special_behavior_button.isChecked() else 'hist_noDC')
        if DICTIONARY_REPLACES_HISTOGRAM:
            objective = objective.replace('hist', 'dict')
        if AUTO_CYCLE_LENGTH_4_PERIODICITY:
            objective = objective.replace('periodicity', 'nonInt_periodicity')

        if 'scribble' in objective and self.canvas.current_display_index == self.canvas.scribble_display_index:
            self.Update_Scribble_Data()
        self.random_inits = ('random' in objective and 'limited' not in objective) or RANDOM_OPT_INITS
        self.multiple_inits = 'random' in objective or MULTIPLE_OPT_INITS
        self.Validate_Z_optimizer(objective)
        self.latest_optimizer_objective = objective
        data = {'LR':self.var_L}
        if self.canvas.Z_optimizer is None:
            print('Initializing Z optimizer...')
            # For the random_l1_limited objective, I want to have L1 differences with respect to the current non-modified image, in case I currently display another image:
            self.canvas.output_image_0_1 = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
            if AUTO_MASK_GRAPHICAL_INPUT and 'scribble' in objective:
                # Experimental: In scribble/imprinting mode while using AUTO_MASK_GRAPHICAL_INPUT, automatically mask out the non-scribbled area within the selected editing region.
                # Apply dilation on the scribble mask, to allow pixels adjacent to the scribble to change, while constraining the rest to remain close to the current output.
                saved_image_mask = 1*self.canvas.HR_selected_mask
                saved_bounding_rect =  1*self.canvas.mask_bounding_rect
                self.canvas.HR_selected_mask = self.canvas.HR_selected_mask * cv2.dilate(self.canvas.current_scribble_mask, np.ones([5, 5]))
                new_bounding_rect = np.array(cv2.boundingRect(np.argwhere(self.canvas.HR_selected_mask)[:,[1,0]]))
                new_bounding_rect[2:] += new_bounding_rect[:2]
                new_bounding_rect = new_bounding_rect/(8 if self.JPEG_GUI else self.opt['scale'])
                new_bounding_rect = np.concatenate([np.floor(new_bounding_rect[:2]),np.ceil(new_bounding_rect[2:])]).astype(int)
                new_bounding_rect[2:] -= new_bounding_rect[:2]
                self.canvas.mask_bounding_rect = new_bounding_rect

            if not np.all(self.canvas.HR_selected_mask) and self.canvas.contained_Z_mask:#Cropping an image region to be optimized, to save on computations and allow adversarial loss
                self.optimizing_region = True
                if objective == 'Adversarial':
                    #Use this D_EXPECTED_LR_SIZE LR_image cropped size when using D that can only work with this size (non mapGAN)
                    gaps = D_EXPECTED_LR_SIZE-self.canvas.mask_bounding_rect[2:]
                    self.bounding_rect_4_opt = np.concatenate([np.maximum(self.canvas.mask_bounding_rect[:2]-gaps//2,0),np.array(2*[D_EXPECTED_LR_SIZE])])
                else:
                    self.bounding_rect_4_opt = np.concatenate([np.maximum(self.canvas.mask_bounding_rect[:2]-MARGINS_AROUND_REGION_OF_INTEREST//2,0),self.canvas.mask_bounding_rect[2:]+MARGINS_AROUND_REGION_OF_INTEREST])
                self.bounding_rect_4_opt[:2] = np.maximum([0,0],np.minimum(self.bounding_rect_4_opt[:2]+self.bounding_rect_4_opt[2:],self.canvas.LR_size[::-1])-self.bounding_rect_4_opt[2:])
                self.bounding_rect_4_opt[2:] = np.minimum(self.bounding_rect_4_opt[:2]+self.bounding_rect_4_opt[2:],self.canvas.LR_size[::-1])-self.bounding_rect_4_opt[:2]
                if self.JPEG_GUI and self.canvas.SR_model.chroma_mode: #In this specific case, bounmding rect values should be even, as I'm going to divide them by 2 at some point
                    self.bounding_rect_4_opt[:2] = self.bounding_rect_4_opt[:2]//2*2
                    self.bounding_rect_4_opt[2:] = 2*np.ceil(self.bounding_rect_4_opt[2:]/2).astype(self.bounding_rect_4_opt.dtype)
                self.MasksStorage(True)
                self.Crop_masks_2_BoundingRect(bounding_rect=self.bounding_rect_4_opt)
                self.Z_mask_4_later_merging = torch.from_numpy(self.canvas.Z_mask).type(self.canvas.output_image_0_1.dtype).to(self.cur_Z.device)
                data['LR'] = self.var_L
                self.canvas.SR_model.Prepare_Input(self.var_L,latent_input=self.cur_Z,compressed_input=True)#Because I'm saving initial Z when initializing optimizer
            else:
                self.optimizing_region = False
            self.iters_per_round = 1*ITERS_PER_OPT_ROUND
            if any([phrase in objective for phrase in ['hist','dict','l12GT']]):
                data['desired'] = [torch.from_numpy(np.ascontiguousarray(np.transpose(hist_im, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0) for hist_im in self.canvas.desired_image]
                if 'l1' in objective and self.optimizing_region:
                    data['desired'] = [self.Crop2BoundingRect(hist_im,self.bounding_rect_4_opt,HR=True) for hist_im in data['desired']]
                data['Desired_Im_Mask'] = self.canvas.desired_image_HR_mask
            elif 'periodicity' in objective:
                for p_num in range(len(self.canvas.periodicity_points)):
                    self.canvas.periodicity_points[p_num] = self.canvas.periodicity_points[p_num]*getattr(self,'periodicity_mag_%d_button'%(p_num+1)).value()/np.linalg.norm(self.canvas.periodicity_points[p_num])
                data['periodicity_points'] = self.canvas.periodicity_points[:2-('1D' in objective)]
            elif 'scribble' in objective:
                data['desired'] = torch.from_numpy(np.ascontiguousarray(np.transpose(self.canvas.image_4_scribbling, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)/255
                data['scribble_mask'] = 1*self.canvas.current_scribble_mask
                if self.JPEG_GUI:
                    data['scribble_mask'] = util.SmearMask2JpegBlocks(data['scribble_mask'])
                data['brightness_factor'] = self.STD_increment.value()#For the brightness increase/decrease functionality
            if any([phrase in objective for phrase in ['STD','Mag']]):
                data['STD_increment'] = self.STD_increment.value()
            initial_Z = 1 * self.cur_Z
            optimization_batch_size = 1
            if self.multiple_inits:
                optimization_batch_size = self.num_random_Zs+(0 if 'random' in objective else 1)
                data['LR'] = data['LR'].repeat([optimization_batch_size,1,1,1]) #When using MULTIPLE_OPT_INITS, optimizing over self.num_random_Zs+1 Zs
                initial_Z = initial_Z.repeat([self.num_random_Zs,1,1,1])
                if 'random' in objective:
                    self.canvas.Z_optimizer_initial_LR = Z_OPTIMIZER_INITIAL_LR_4_RANDOM
                    if not HIGH_OPT_ITERS_LIMIT:
                        self.iters_per_round *= 3
                    if 'limited' in objective:
                        data['rmse_weight'] = 1*self.randomLimitingWeightBox.value() if self.randomLimitingWeightBox_Enabled else 1
                else: # this means we are not in 'random' objective but using MULTIPLE_OPT_INITS, so we want one Z initialized to current, while the rest should be random. We indicate this by passing a single Z
                    initial_Z = initial_Z[0].unsqueeze(0)
            if HIGH_OPT_ITERS_LIMIT:#Signaling the optimizer to keep iterating until loss stops decreasing, using iters_per_round as window size. Still have some fixed limit.
                self.iters_per_round *= -1
            if self.canvas.Z_optimizer_logger is None:
                self.canvas.Z_optimizer_logger = []
                for i in range(optimization_batch_size):
                    self.canvas.Z_optimizer_logger.append(Logger(self.canvas.opt,tb_logger_suffix='_%s%s'%(objective,'_%d'%(i) if self.multiple_inits else '')))
            if self.JPEG_GUI:
                data['Comp'] = data['LR']
                data['QF'] = self.canvas.SR_model.jpeg_extractor.QF.view(1)
                data['chroma_input'] = self.chroma_input
            self.canvas.Z_optimizer = Z_optimizer(objective=objective,Z_size=[val*self.canvas.SR_model.Z_size_factor for val in data['LR'].size()[2:]],model=self.canvas.SR_model,
                Z_range=self.max_SVD_Lambda,data=data,initial_LR=self.canvas.Z_optimizer_initial_LR,loggers=self.canvas.Z_optimizer_logger,max_iters=self.iters_per_round,
                image_mask=self.canvas.HR_selected_mask,Z_mask=self.canvas.Z_mask,auto_set_hist_temperature=self.auto_set_hist_temperature,
                batch_size=optimization_batch_size,random_Z_inits=self.random_inits,initial_Z=initial_Z,
                jpeg_extractor=self.canvas.SR_model.jpeg_extractor if self.JPEG_GUI else None,non_local_Z_optimization=NON_LOCAL_Z_OPTIMIZATION)
            if self.optimizing_region:
                self.MasksStorage(False)
            if AUTO_MASK_GRAPHICAL_INPUT and 'scribble' in objective:
                # Restoring the saved image mask:
                self.canvas.HR_selected_mask,self.canvas.mask_bounding_rect = 1*saved_image_mask,1*saved_bounding_rect

        elif 'random' in objective:
            self.canvas.Z_optimizer.cur_iter = 0
        num_looping_iters = 30 if loop else 1
        reset_Z_optimizer,decrease_LR = False,False
        optimization_start_time = time.time()
        for mini_epoch in range(num_looping_iters):
            self.stored_Z = 1 * self.cur_Z
            if self.multiple_inits:
                self.stored_masked_zs = 1*self.canvas.random_Zs
                if 'random' not in objective:
                    self.stored_masked_zs = torch.cat([1*self.cur_Z,self.stored_masked_zs],0)
            else:
                self.stored_masked_zs = 1 * self.cur_Z # Storing previous Z for two reasons: To recover the big picture Z when optimizing_region, and to recover previous Z if loss did not decrease
            try:
                self.statusBar.showMessage('Optimizing for %s...'%(objective))
                optimization_failed =True
                self.cur_Z = self.canvas.Z_optimizer.optimize()
                optimization_failed = False
            except Exception as e:
                self.statusBar.showMessage('%s optimization failed: %s' % (objective,e), ERR_MESSAGE_DURATION)
                print('Optimization failed: ',e)
                if 'loss' in self.canvas.Z_optimizer.__dict__.keys() and 'bins' in self.canvas.Z_optimizer.loss.__dict__.keys():
                    print('# desired hist images: %d'%(len(self.canvas.desired_image)))
                    print('# Bins: %d, # Image patches: %d'%(self.canvas.Z_optimizer.loss.bins.size(-1),self.canvas.Z_optimizer.loss.patch_extraction_mat.size(1)))
            discard_result = optimization_failed or self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1] < 0
            time_limit_reached = time.time()-optimization_start_time>Z_OPTIMIZATION_TIME_LIMIT
            if discard_result:
                self.cur_Z = 1 * self.stored_Z
                self.canvas.SR_model.Prepare_Input(self.var_L,latent_input=self.cur_Z.type(self.var_L.type()),compressed_input=True)
                self.SelectImage2Display()
                reset_Z_optimizer = True
                decrease_LR = not optimization_failed
                if loop:
                    break
            else:
                if self.optimizing_region:
                    temp_Z = (1 * self.cur_Z).to(self.stored_masked_zs.device)
                    self.cur_Z = 1 * self.stored_masked_zs
                    cropping_rect = 1*self.bounding_rect_4_opt
                    if self.canvas.HR_Z:
                        cropping_rect = [self.canvas.H_L_domains_ratio*val for val in self.bounding_rect_4_opt]
                    for Z_num in range(temp_Z.size(0)): # Doing this for the case of finding random Zs far from one another:
                        if ONLY_MODIFY_MASKED_AREA_WHEN_OPTIMIZING:
                            self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]] =\
                                self.Z_mask_4_later_merging*temp_Z[Z_num, ...]+\
                                (1-self.Z_mask_4_later_merging)*self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]]
                        else:
                            self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]] = temp_Z[Z_num,...]
                avoid_Z_undo_list_update = mini_epoch<(num_looping_iters-1) and not time_limit_reached
                if self.multiple_inits:
                    self.canvas.random_Zs = 1*self.cur_Z[-self.num_random_Zs:]
                    self.Process_Z_Alternatives()
                    if 'random' in objective:
                        self.cur_Z = 1*self.stored_Z
                    else:
                        self.cur_Z = 1 * self.cur_Z[0].unsqueeze(0)
                        display_index = ([self.cur_Z_im_index]+self.random_display_indexes)[np.argmin(self.canvas.Z_optimizer.latest_Z_loss_values)]
                        print('Loss values (%d):'%(np.argmin(self.canvas.Z_optimizer.latest_Z_loss_values)),['%.3e'%(val) for val in self.canvas.Z_optimizer.latest_Z_loss_values])
                        self.ReProcess(chosen_display_index=display_index,dont_update_undo_list=avoid_Z_undo_list_update)
                else:
                    self.DeriveControlValues()
                    self.ReProcess(chosen_display_index=self.cur_Z_im_index if 'scribble' in objective else None,dont_update_undo_list=avoid_Z_undo_list_update)
            if loop and time_limit_reached:
                print('Z optimization time limit (%d sec.) reached.'%(Z_OPTIMIZATION_TIME_LIMIT))
                break

            if not optimization_failed:
                print('%d: LR=%.1e, %d iterations: %s loss decreased from %.2e to %.2e by %.2e (factor of %.2e)' % (mini_epoch,self.canvas.Z_optimizer.LR,len(self.canvas.Z_optimizer.loss_values), self.canvas.Z_optimizer.objective,
                    self.canvas.Z_optimizer.loss_values[0],self.canvas.Z_optimizer.loss_values[-1],self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1],
                    self.canvas.Z_optimizer.loss_values[-1]/self.canvas.Z_optimizer.loss_values[0]))
                if (self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))]-self.canvas.Z_optimizer.loss_values[-1])/\
                        np.abs(self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))])<1e-2*self.canvas.Z_optimizer_initial_LR: #If the loss did not decrease, I decrease the optimizer's learning rate
                    decrease_LR = True
                    # self.canvas.Z_optimizer_initial_LR /= 5
                    # print('Loss decreased too little relative to beginning, decreasing learning rate to %.3e'%(self.canvas.Z_optimizer_initial_LR))
                    # reset_Z_optimizer = True
                    if loop:
                        print('Breaking optimization loop')
                        if mini_epoch<(num_looping_iters-1):
                            self.Add_Z_2_history()
                        break
                else: # This means I'm happy with this optimizer (and its learning rate), so I can cancel the auto-hist-temperature setting, in case it was set to True.
                    if self.auto_hist_temperature_mode_Enabled:
                        self.auto_set_hist_temperature = False
                        self.auto_hist_temperature_mode_button.setChecked(False)
        if reset_Z_optimizer:
            self.canvas.Z_optimizer = None
            # Optimization already caused backpropagation through the current fake_H without recomputing it. So it needs to be recomputed to allow another backpropagation:
            self.Feed_n_Run_model(self.cur_Z)
        if decrease_LR:
            self.canvas.Z_optimizer_initial_LR /= 5
            print('Loss decreased too little relative to beginning, decreasing learning rate to %.3e' % (
                self.canvas.Z_optimizer_initial_LR))
        if not optimization_failed:
            self.statusBar.showMessage('%s optimization is done.' % (objective), INFO_MESSAGE_DURATION)

    def DeriveControlValues(self):
        normalized_Z = self.Z_2_three_channels(1*self.cur_Z.squeeze(0))
        normalized_Z = (normalized_Z+self.max_SVD_Lambda)/2/self.max_SVD_Lambda
        normalized_Z[2] /= 2
        new_control_values = torch.stack(util.SVD_Symmetric_2x2(*normalized_Z),0).to( self.canvas.control_values)# Lambda values are not guarnteed to be in [0,self.max_SVD_Lambda], despite Z being limited to [-self.max_SVD_Lambda,self.max_SVD_Lambda].
        self.canvas.derived_controls_indicator = np.logical_or(self.canvas.derived_controls_indicator,self.canvas.Z_mask)
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to( self.canvas.control_values)
        self.canvas.control_values = Z_mask*new_control_values+(1-Z_mask)*self.canvas.control_values
        self.canvas.Update_Z_Sliders()

    def Clear_Z_Mask(self):
        self.canvas.Z_mask = np.ones(self.canvas.Z_size)
        self.canvas.HR_selected_mask = np.ones(self.canvas.HR_size)
        self.canvas.update_HR_mask_display_size()
        if 'current_scribble_mask' in self.canvas.__dict__.keys():
            self.canvas.safe_scribble_apply_buttons_enabling()
        self.canvas.LR_mask_vertices = [(0,0),tuple(self.canvas.LR_size[::-1])]
        self.canvas.HR_mask_vertices = [(0,0),tuple(self.canvas.HR_size[::-1])]
        self.canvas.update_mask_bounding_rect()
        self.canvas.Update_Z_Sliders()
        self.canvas.Z_optimizer_Reset()
        self.canvas.selection_display_timer_cleanup(clear_data=True)

    def Invert_Z_Mask(self):
        self.canvas.Z_mask = 1-self.canvas.Z_mask
        self.canvas.update_HR_mask_display_size()
        self.canvas.HR_selected_mask = 1-self.canvas.HR_selected_mask
        self.canvas.Z_optimizer_Reset()
        self.canvas.contained_Z_mask = not self.canvas.contained_Z_mask
        if not self.JPEG_GUI:
            if self.canvas.contained_Z_mask:
                self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.mask_bounding_rect[2:]]))
            else:
                self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.HR_size]))

    def ApplyUniformZ(self):
        self.canvas.Update_Z_Sliders()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.canvas.control_values.dtype).to(self.canvas.control_values.device)
        if False and self.canvas.SR_model.num_latent_channels!=3:
            uniform_values_to_assign = ((self.canvas.control_values*Z_mask.unsqueeze(0)).sum(-1).sum(-1)/Z_mask.sum()).view(-1,1,1)
        else:
            uniform_values_to_assign = torch.from_numpy(self.canvas.previous_sliders_values).type(Z_mask.dtype).to(Z_mask.device)
        self.canvas.control_values = Z_mask * uniform_values_to_assign + (1 - Z_mask) * self.canvas.control_values
        self.canvas.Z_optimizer_Reset()
        self.Recompose_cur_Z()
        self.ReProcess()
        self.canvas.derived_controls_indicator = (self.canvas.Z_mask*0+(1-self.canvas.Z_mask)*self.canvas.derived_controls_indicator).astype(np.bool)

    def Recompose_cur_Z(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        new_Z = self.canvas.control_values.unsqueeze(0)
        new_Z = util.SVD_2_LatentZ(new_Z,max_lambda=self.max_SVD_Lambda).to(self.cur_Z.device)
        new_Z = self.Repeat_Z_3_channels(new_Z)
        self.cur_Z = Z_mask * new_Z + (1 - Z_mask) * self.cur_Z

    def SetZ_And_Display(self,value,index,dont_update_undo_list=False):
        self.SetZ(value,index)
        self.Recompose_cur_Z()
        self.ReProcess(dont_update_undo_list=dont_update_undo_list)

    def SetZ(self,value,index,reset_optimizer=True):
        if reset_optimizer:
            self.canvas.Z_optimizer_Reset()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
        derived_controls_indicator = torch.from_numpy(self.canvas.derived_controls_indicator).type(self.cur_Z.dtype)
        value_increment = value - 1*self.canvas.previous_sliders_values[index]
        additive_values = torch.from_numpy(value_increment).type(self.canvas.control_values.dtype) + self.canvas.control_values[index].to(derived_controls_indicator.device)
        masked_new_values = Z_mask *(value * (1 - derived_controls_indicator) + derived_controls_indicator * additive_values)
        self.canvas.control_values[index] =  masked_new_values + (1 - Z_mask) * self.canvas.control_values[index].to(derived_controls_indicator.device)
        self.canvas.previous_sliders_values[index] = (1-self.canvas.Z_mask)*self.canvas.previous_sliders_values[index]+self.canvas.ReturnMaskedMapAverage(self.canvas.control_values[index].data.cpu().numpy())

    def Update_Default_Z_Image(self):
        if 'random_Z_images' in self.canvas.__dict__.keys():
            self.canvas.random_Z_images[0] = 1 * self.canvas.output_image_0_1[0]
        else:
            self.canvas.random_Z_images = torch.cat([1*self.canvas.output_image_0_1,torch.zeros_like(self.canvas.output_image_0_1).repeat([self.num_random_Zs,1,1,1])],0)

    def ToggleDisplay_Z_Image(self,checked):
        for mode in self.canvas.scribble_modes:
            getattr(self,'%s_button'%(mode)).setEnabled(not checked)
        self.Update_Image_Display()

    def Change_Display_Zoom(self,increase=None,reset=False):
        # Save scribble data and mask currently on canvas before changing canvas size:
        self.Update_Scribble_Data()
        # Change display zoom factor:
        if reset:
            self.canvas.display_zoom_factor = 1*DISPLAY_ZOOM_FACTOR
        else:
            self.canvas.display_zoom_factor += 1 if increase else -1
        self.canvas.cur_loc_step_size = self.canvas.display_zoom_factor // (2 * self.canvas.display_zoom_factor)
        self.Safely_Enable_Zoom_Buttons(True)
        # Resize saved data whose dimensions correspond to display size:
        self.canvas.image_4_scribbling_display_size = \
            util.ResizeScribbleImage(self.canvas.image_4_scribbling_display_size,dsize=tuple([self.canvas.display_zoom_factor * val for val in self.canvas.HR_size]))
        self.Update_Scribble_Mask_Canvas(initialize=True)
        self.canvas.update_HR_mask_display_size()
        # Apply new size image to canvas:
        self.Update_Image_Display()
        # Update canvas window size and title:
        self.Update_Canvas_Size_and_Title()

    def Safely_Enable_Zoom_Buttons(self,enable):
        self.DecreaseDisplayZoom_button.setEnabled(enable and self.canvas.display_zoom_factor>DISPLAY_ZOOM_FACTORS_RANGE[0])
        self.IncreaseDisplayZoom_button.setEnabled(enable and self.canvas.display_zoom_factor <DISPLAY_ZOOM_FACTORS_RANGE[1])

    def Update_Canvas_Size_and_Title(self,canvas_pos=None):
        if canvas_pos is None:
            canvas_pos = self.canvas.geometry().getCoords()[:2]
        self.canvas.setGeometry(QRect(canvas_pos[0], canvas_pos[1], self.canvas.display_zoom_factor * self.canvas.HR_size[1],
                  self.canvas.display_zoom_factor * self.canvas.HR_size[0]))
        self.canvas.setWindowTitle(self.image_name+(' (x%d)'%(self.canvas.display_zoom_factor) if self.canvas.display_zoom_factor>1 else ''))
        self.canvas.show()

    def Update_Image_Display(self):
        self.canvas.selection_display_timer_cleanup() #Removing selection painting before changing the pixmap, as it cannot be removed (using XOR) afterwords.
        pixmap = QPixmap()
        if self.Zdisplay_button.isChecked():
            im_2_display = (255/2/self.max_SVD_Lambda*(self.max_SVD_Lambda+self.Z_2_display[0].data.cpu().numpy().transpose(1,2,0)).copy())
            if list(im_2_display.shape[:2])!=self.canvas.HR_size: # Should be satisifeid when JPEG GUI:
                if im_2_display.shape[2]==3:
                    im_2_display = util.ResizeScribbleImage(im_2_display,tuple(self.canvas.HR_size))
                elif im_2_display.shape[2] == 8**2:
                    im_2_display = util.Z_64channels2image(im_2_display)
                else:
                    raise Exception('Unsupported # of Z channels for display')
        else:
            if self.canvas.current_display_index==self.canvas.scribble_display_index:
                im_2_display = 1*self.canvas.image_4_scribbling_display_size
            else:
                im_2_display = 255 * self.canvas.output_image_0_1[0].detach().float().cpu().numpy().transpose(1, 2, 0).copy()
        if self.canvas.display_zoom_factor>1 and not ((not self.Zdisplay_button.isChecked()) and self.canvas.current_display_index==self.canvas.scribble_display_index):
            # For the specific case of updating scribble image, image is allready in correct size
            im_2_display = imresize(im_2_display,self.canvas.display_zoom_factor)
        pixmap.convertFromImage(qimage2ndarray.array2qimage(im_2_display))
        self.canvas.setPixmap(pixmap)
        if self.canvas.selection_display and self.canvas.current_display_index!=self.canvas.scribble_display_index:
            # Recreating the selection painting if a selection is in place. Not showing the selection when in scribble mode, as it will be added to the scribble wherever it crosses the selection painting.
            self.canvas.selection_display_timer_creation()
        if DISPLAY_INDUCED_LR:
            self.Update_LR_Display()

    def Update_LR_Display(self):
        pixmap = QPixmap()
        pixmap.convertFromImage(qimage2ndarray.array2qimage(255 * self.induced_LR_image[0].data.cpu().numpy().transpose(1,2,0).copy()))
        self.LR_canvas.setPixmap(pixmap)

    def ReProcess(self,chosen_display_index=None,dont_update_undo_list=False):
        self.Compute_SR_Image()
        self.Update_Default_Z_Image()
        if not dont_update_undo_list:
            self.Add_Z_2_history()
        self.SelectImage2Display(chosen_index=chosen_display_index)
    #
    def Load_Z_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Z file to deduce editing mask", "","PNG image files (*_Z.png)")
        if path:
            loaded_Z = data_util.read_img(None, path)
            edited_pixels_map = np.any(loaded_Z!=127/255,axis=2)
            self.canvas.HR_selected_mask = 1*edited_pixels_map
            self.canvas.FoolAdversary_button.setEnabled(False)
            self.canvas.contained_Z_mask = False
            assert self.canvas.HR_Z,'Not supprting other option'
            self.canvas.Z_mask = 1*edited_pixels_map
            self.canvas.update_HR_mask_display_size()
            self.canvas.Update_Z_Sliders()
            self.canvas.Z_optimizer_Reset()
            self.canvas.safe_scribble_apply_buttons_enabling()

    def Assign_Q_Table(self,QF_or_table=None):
        if self.real_JPEG_image:
        #     Ignoring given QF:
            QF = False
            self.QF_box.setEnabled(False)
            QF_or_table = self.Q_Tables.copy()
        else:
            QF = True
            self.QF_box.setEnabled(True)
            self.QF_box.setMaximum(MAXIMAL_JPEG_QF)
            QF_or_table = torch.tensor(QF_or_table)
        self.canvas.SR_model.jpeg_compressor.Set_Q_Table(QF_or_table,QF)
        self.canvas.SR_model.jpeg_extractor.Set_Q_Table(QF_or_table,QF)
        self.canvas.SR_model.jpeg_compressor_Y.Set_Q_Table(QF_or_table,QF)
        self.canvas.SR_model.jpeg_extractor_Y.Set_Q_Table(QF_or_table,QF)
        self.canvas.SR_model.jpeg_compressor_Y_non_quantized.Set_Q_Table(QF_or_table,QF)
        self.canvas.SR_model.jpeg_compressor_non_quantized.Set_Q_Table(QF_or_table,QF)
        if self.real_JPEG_image:
            self.QF_box.setMaximum(1000)
            self.QF_box.setValue(self.canvas.SR_model.jpeg_compressor_Y.QF)
        else:
            self.Q_Tables = [T for T in self.canvas.SR_model.jpeg_extractor.Q_table.squeeze()[:2].data.cpu().numpy()]

    def Update_QF(self,QF):
        self.Assign_Q_Table(QF)
        self.var_L = self.canvas.SR_model.jpeg_compressor_Y(self.loaded_image[:,0,...].unsqueeze(1))
        self.input_image = torch.clamp(util.Tensor_YCbCR2RGB(self.canvas.SR_model.Return_Compressed(self.loaded_image)/ 255) , 0, 1)
        self.ReProcess()

    def Initialize_SR_model(self,kernel=None,reprocess=True):
        if self.JPEG_GUI:
            self.canvas.SR_model = create_model(self.opt, chroma_mode=True)
            self.canvas.SR_model.jpeg_compressor_Y_non_quantized = JPEG(compress=True,chroma_mode=False, downsample_and_quantize=False,block_size=8)\
                .to(self.canvas.SR_model.device)
            self.canvas.SR_model.jpeg_compressor_non_quantized = JPEG(compress=True,chroma_mode=True, downsample_and_quantize=False,block_size=self.opt['scale'])\
                .to(self.canvas.SR_model.device)
        else:
            self.canvas.SR_model = create_model(self.opt, init_Dnet=False, init_Fnet=VGG_RANDOM_DOMAIN,kernel=kernel)
        self.canvas.Z_optimizer_Reset()
        if reprocess:
            self.ReProcess()

    def Load_Z(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Z file", "","PNG image files (*_Z.png)")
        if path:
            loaded_Z = data_util.read_img(None, path)
            if self.JPEG_GUI:
                assert list(loaded_Z.shape[:2]) == [8*v for v in self.canvas.LR_size], 'Size of Z doesn''t match...'
                loaded_Z = loaded_Z.reshape([self.canvas.LR_size[0],8,self.canvas.LR_size[1],8]).transpose(0,2,1,3).reshape(self.canvas.LR_size+[64])
                # np.reshape(Z, list(Z.shape[:2]) + [8, 8]).transpose((0, 2, 1, 3)).reshape(list(8 * np.array(Z.shape[:2])) + [1])
            else:
                loaded_Z = loaded_Z[:, :, [2, 1, 0]]
                assert list(loaded_Z.shape[:2])==self.canvas.Z_size,'Size of Z does not match image size'
            self.cur_Z = torch.from_numpy(np.transpose(2*self.max_SVD_Lambda*loaded_Z-self.max_SVD_Lambda, (2, 0, 1))).float().to(self.cur_Z.device).type(self.cur_Z.dtype).unsqueeze(0)
            self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1])
            self.ReProcess()
            stored_mask = 1*self.canvas.Z_mask
            self.canvas.Z_mask = np.ones_like(self.canvas.Z_mask)
            self.DeriveControlValues()
            self.canvas.derived_controls_indicator = self.Estimate_DerivedControlIndicator()
            scribble_data_path = path.replace('_Z.png','_scribble_data.npz')
            if os.path.exists(scribble_data_path):
                loaded_data = np.load(scribble_data_path)
                if 'estimated_kernel' in loaded_data.files:
                    self.estimated_kernel = loaded_data['estimated_kernel']
                    self.Change_kernel_in_use(use_estimated=True)
                if 'scribble_image' in loaded_data.files:
                    self.canvas.image_4_scribbling = loaded_data['scribble_image']
                    self.canvas.image_4_scribbling_display_size = 1 * self.canvas.image_4_scribbling
                    if self.canvas.display_zoom_factor > 1:
                        self.canvas.image_4_scribbling_display_size = util.ResizeScribbleImage(self.canvas.image_4_scribbling_display_size,
                                                                                  dsize=tuple([self.canvas.display_zoom_factor*val for val in self.canvas.HR_size]))
                    self.Initialize_Image_4_Scribbling_Display_Size()
                    self.canvas.current_scribble_mask = loaded_data['scribble_mask']
                    self.Update_Scribble_Mask_Canvas(initialize=True)
                    self.canvas.safe_scribble_apply_buttons_enabling()

            self.canvas.Z_mask = 1*stored_mask
            self.canvas.update_HR_mask_display_size()
            self.canvas.Z_optimizer_Reset()

    def Estimate_DerivedControlIndicator(self):
        PATCH_SIZE_4_ESTIMATION = 3
        patch_extraction_map = ReturnPatchExtractionMat(self.canvas.Z_mask,PATCH_SIZE_4_ESTIMATION,device=self.cur_Z.device,patches_overlap=1)
        STD_map = torch.sparse.mm(patch_extraction_map, self.cur_Z.mean(dim=1).view([-1, 1])).view([PATCH_SIZE_4_ESTIMATION ** 2, -1]).std(dim=0).view(
            [val-PATCH_SIZE_4_ESTIMATION+1 for val in list(self.cur_Z.size()[2:])])
        return np.pad((STD_map>0).data.cpu().numpy().astype(np.bool),pad_width=int(PATCH_SIZE_4_ESTIMATION//2),mode='edge')

    def Z_2_three_channels(self,multi_channel_Z):
        if self.JPEG_GUI:
            return self.canvas.SR_model.Z_2_three_channels(multi_channel_Z)
        else:
            return multi_channel_Z

    def Repeat_Z_3_channels(self,Z):
        if self.JPEG_GUI:
            return self.canvas.SR_model.Repeat_Z_3_channels(Z)
        else:
            return Z

    def Decode_JPEG_input(self,path):
        DIRECTLY_LOAD_DCT_COEFFS = True
        if DIRECTLY_LOAD_DCT_COEFFS:
            # This does not work as expected yet. Two major issues:
            # 1.The DCT coeffs and the resulting images range is very small, approximately 255/3 smaller (that's whay I apply the approximate correction)
            # 2.For some images, e.g. /home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/HR_images/15-tufted-duck.w700.h467.jpg, the unnormalized DCT coeffs do not match the normalized ones, when multiplied by the Quantization tables.
            LOAD_NORMALIZED = True
            def color_DCT_2_16pix_block_DCT_tensor(input_DCT):
                input_shape = input_DCT.shape[:2]
                output_tensor = 255/3*input_DCT.astype(np.int32)
                # output_tensor = 1*input_DCT
                if LOAD_NORMALIZED:
                    output_tensor /= self.Q_Tables[1].reshape([1,1,64])
                assert np.all(output_tensor.round() == output_tensor), 'Expected the DCT coefficients divided by the saved Q-table to be integers'
                output_tensor = output_tensor.reshape(input_shape[0],input_shape[1],1,8,1,8)
                output_tensor = np.pad(output_tensor,((0,0),(0,0),(0,1),(0,0),(0,1),(0,0)))
                output_tensor = output_tensor.reshape(input_shape[0],input_shape[1],256)
                return torch.from_numpy(output_tensor.transpose((2,0,1))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            dct_y, dct_cb, dct_cr = jpeg_load(path,normalized=LOAD_NORMALIZED)
            assert np.abs(np.array(dct_y.shape[:2])/2-np.array(dct_cb.shape[:2])).max()<=1/2,'Assuming chroma is downsampled in half in both axes'
            for axis_num in range(2):
                while np.any(dct_y.shape[axis_num]!=2*dct_cb.shape[axis_num]):
                    if dct_y.shape[axis_num]>2*dct_cb.shape[axis_num]:
                        dct_y = dct_y.take(np.arange(dct_y.shape[axis_num]-1),axis_num)
                    else:
                        dct_cb = dct_cb.take(np.arange(dct_cb.shape[axis_num] - 1), axis_num)
                        dct_cr = dct_cr.take(np.arange(dct_cr.shape[axis_num] - 1), axis_num)

            dct_y_input2model = 255/3*dct_y.astype(np.int32).transpose((2,0,1))
            # dct_y_input2model = dct_y.transpose((2,0,1))
            if LOAD_NORMALIZED:
                dct_y_input2model /= self.Q_Tables[0].reshape([64,1,1])
            assert np.all(dct_y_input2model.round()==dct_y_input2model),'Expected the DCT coefficients divided by the saved Q-table to be integers'
            self.real_JPEG_image = True
            self.Assign_Q_Table()
            self.var_L = torch.from_numpy(dct_y_input2model).float().to(self.canvas.SR_model.device).unsqueeze(0)
            self.chroma_input = self.canvas.SR_model.jpeg_extractor(torch.cat([color_DCT_2_16pix_block_DCT_tensor(dct_cb),color_DCT_2_16pix_block_DCT_tensor(dct_cr)],1))
            self.loaded_image = torch.cat([self.canvas.SR_model.jpeg_extractor_Y(self.var_L),self.chroma_input],1)
        else:
            self.real_JPEG_image = True
            self.Assign_Q_Table()
            loaded_rgb = data_util.read_img(None, path)
            if loaded_rgb.ndim>2 and loaded_rgb.shape[2] == 3:
                loaded_rgb = loaded_rgb[:, :, [2, 1, 0]]
            loaded_rgb *= 255
            loaded_YCbCr = rgb2ycbcr(1*loaded_rgb,only_y=False)
            loaded_YCbCr = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_YCbCr, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            loaded_YCbCr = loaded_YCbCr[:,:,:loaded_YCbCr.size(2)//16*16,:loaded_YCbCr.size(3)//16*16]
            self.var_L = self.canvas.SR_model.jpeg_compressor_Y(loaded_YCbCr[:,0,...].unsqueeze(1))
            chroma_input = self.canvas.SR_model.jpeg_extractor(self.canvas.SR_model.jpeg_compressor(loaded_YCbCr))[:,1:,...]
            loaded_image = torch.cat([self.canvas.SR_model.jpeg_extractor_Y(self.var_L),chroma_input],1)
            loaded_image = torch.clamp(util.Tensor_YCbCR2RGB(loaded_image),0,255).data[0].cpu().numpy().transpose((1,2,0))
            mean_jpeg_deciphering_error = np.abs(loaded_image-loaded_rgb).mean()
            self.statusBar.showMessage('Mean error %.3f gray-levels when deciphering JPEG coefficients'%(mean_jpeg_deciphering_error),INFO_MESSAGE_DURATION)
            loaded_image = rgb2ycbcr(loaded_image,only_y=False)
            self.loaded_image = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            self.chroma_input = self.loaded_image[:,1:,...]

    def open_file(self,path=None,HR_image=False,canvas_pos=None):
        """
        Open image file for editing, scaling the smaller dimension and cropping the remainder.
        :return:
        """
        loaded_image = None
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self,"Open HR file to be downsampled" if HR_image else "Open file", "",
                'JPEG image files (*.jpg)' if (self.JPEG_GUI and not HR_image) else IMAGE_FILE_EXST_FILTER)
        if not path or not os.path.isfile(path):
            print('Not loading any image')
            return
        self.image_name = path.split('/')[-1].split('.')[0]
        loaded_image = data_util.read_img(None, path)
        if loaded_image.ndim>2 and loaded_image.shape[2] == 3:
            loaded_image = loaded_image[:, :, [2, 1, 0]]
        elif loaded_image.ndim<3 or loaded_image.shape[2]==1:
            loaded_image = np.tile(loaded_image.reshape([loaded_image.shape[0],loaded_image.shape[1],1]),[1,1,3])
        if HR_image:
            modulu_size = self.canvas.opt['scale']
            loaded_image = loaded_image[:loaded_image.shape[0]//modulu_size*modulu_size,:loaded_image.shape[1]//modulu_size*modulu_size,:] #Removing bottom right margins to make the image shape adequate to this SR factor
            self.GT_HR = torch.from_numpy(np.transpose(loaded_image, (2, 0, 1))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            self.canvas.HR_size = list(self.GT_HR.size()[2:])
            if self.JPEG_GUI:
                loaded_image = 255 * rgb2ycbcr(loaded_image, only_y=False)
            loaded_image = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            if self.JPEG_GUI:
                self.real_JPEG_image = False
                self.QF_box.setValue(10)
                self.Assign_Q_Table(10)
                self.loaded_image = loaded_image
                self.var_L = self.canvas.SR_model.jpeg_compressor_Y(self.loaded_image[:,0,...].unsqueeze(1))
                self.chroma_input = self.loaded_image[:,1:,...]
            else:
                self.var_L = self.canvas.SR_model.netG.module.DownscaleOP(loaded_image)
                self.LR_image = np.clip(255*self.var_L[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8)
            self.DisplayedImageSelection_button.model().item(self.GT_HR_index).setEnabled(True)
        else:
            if self.JPEG_GUI:
                try:
                    self.Q_Tables = {int(k): util.zigzag_list_2_Q_table(list(v)).astype(np.float32) for k, v in PIL_Image.open(path).quantization.items()}
                    self.Decode_JPEG_input(path)
                    self.canvas.HR_size = list(self.loaded_image.size())[2:]
                except Exception as e:
                    self.statusBar.showMessage('%s loading failed: %s' % (path.split('/')[-1], e), ERR_MESSAGE_DURATION)
                    print('%s loading failed: %s' % (path.split('/')[-1], e))
            else:
                self.var_L = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)
                self.LR_image = (255*loaded_image).astype(np.uint8)
                self.canvas.HR_size = [self.opt['scale']*v for v in self.LR_image.shape[:2]]
            self.DisplayedImageSelection_button.model().item(self.GT_HR_index).setEnabled(False)
        if self.JPEG_GUI:
            self.input_image = torch.clamp(util.Tensor_YCbCR2RGB(self.canvas.SR_model.Return_Compressed(self.loaded_image)/255),0, 1)
        else:
            self.input_image = torch.from_numpy(np.transpose(cv2.resize(self.LR_image, dsize=tuple(self.canvas.HR_size[::-1]),
                                      interpolation=cv2.INTER_NEAREST)/255,(2,0,1))).float().unsqueeze(0)
        if self.display_ESRGAN:
            ESRGAN_opt = option.parse('./options/test/GUI_esrgan.json', is_train=False,name='RRDB_ESRGAN_x4')
            ESRGAN_opt['name']
            ESRGAN_opt = option.dict_to_nonedict(ESRGAN_opt)
            ESRGAN_opt['network_G']['latent_input'] = 'None'
            ESRGAN_opt['network_G']['latent_channels'] = 0
            ESRGAN_opt['network_G']['CEM_arch'] = 0
            ESRGAN_model = create_model(ESRGAN_opt)
            ESRGAN_model.netG.eval()
            with torch.no_grad():
                self.ESRGAN_SR = ESRGAN_model.netG(self.var_L).detach().to(torch.device('cpu'))
            self.canvas.HR_size = list(self.ESRGAN_SR.size()[2:])

        if 'random_Z_images' in self.canvas.__dict__.keys():
            del self.canvas.random_Z_images
        self.canvas.LR_size = list(self.var_L.size()[2:])
        if self.JPEG_GUI:
            self.canvas.Z_size = self.canvas.LR_size
        else:
            self.canvas.Z_size = [val*self.canvas.H_L_domains_ratio for val in self.canvas.LR_size] if self.canvas.HR_Z else self.canvas.LR_size
        self.canvas.Z_mask = np.ones(self.canvas.Z_size)
        self.canvas.HR_selected_mask = np.ones(self.canvas.HR_size)
        self.canvas.update_HR_mask_display_size()
        self.canvas.derived_controls_indicator = np.zeros(self.canvas.Z_size)
        self.cur_Z = torch.zeros(size=[1,self.canvas.SR_model.num_latent_channels]+self.canvas.Z_size).to(self.canvas.SR_model.device)
        self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1]).to(self.cur_Z.device)
        self.canvas.previous_sliders_values = \
            np.array([self.canvas.Z0_slider.value(),self.canvas.Z1_slider.value(),self.canvas.third_channel_slider.value()]).reshape([3,1,1])/100*np.ones([1]+self.canvas.Z_size)
        # self.canvas.control_values = 0.5*torch.ones_like(self.cur_Z).squeeze(0)
        self.canvas.control_values = self.Z_2_three_channels(0.5*torch.ones_like(self.cur_Z).squeeze(0))
        self.SetZ(0.5*self.max_SVD_Lambda, 0)
        self.SetZ(0.5*self.max_SVD_Lambda, 1)
        self.SetZ(0.5, 2)
        self.Recompose_cur_Z()
        # Reset some stuff:
        if self.using_estimated_kernel: # Need to reset model to default kernel:
            self.using_estimated_kernel = False
            self.estimatedKenrel_button.setChecked(False)
        self.estimatedKenrel_button.setEnabled((not HR_image) and (self.canvas.opt['scale'] in [2,4])) #KernelGAN only supprot 2x and 4x SR. For synthetically downscaled HR images, there is no need to estimated the kernel.
        self.estimated_kernel = None
        self.Z_history = deque(maxlen=Z_HISTORY_LENGTH)
        self.Z_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
        self.canvas.scribble_history = deque(maxlen=Z_HISTORY_LENGTH)
        self.canvas.scribble_mask_history = deque(maxlen=Z_HISTORY_LENGTH)
        self.canvas.scribble_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
        self.canvas.scribble_mask_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
        self.canvas.current_display_index = 1*self.cur_Z_im_index
        if self.JPEG_GUI:
            self.canvas.Z_optimizer_Reset()
            self.ReProcess()
        else:
            self.Initialize_SR_model(kernel='reset_2_default')
        self.saved_outputs_counter = 0
        self.DisplayedImageSelection_button.setCurrentIndex(self.cur_Z_im_index)
        self.no_Z_image = torch.from_numpy(np.transpose(2 * (resize(image=data_util.read_img(None,
            'icons/X.png')[:, :, ::-1], output_shape=self.canvas.HR_size) - 0.5), (2, 0, 1))).float().to(self.canvas.SR_model.device).unsqueeze(0)
        if 'current_scribble_mask' in self.canvas.__dict__.keys():
            del self.canvas.current_scribble_mask
        self.Clear_Z_Mask()
        self.canvas.image_4_scribbling = None
        self.Reset_Image_4_Scribbling()
        # self.canvas.show()
        self.Change_Display_Zoom(reset=True)
        self.Update_Canvas_Size_and_Title(canvas_pos)

    def Add_Z_2_history(self,clear_redo_list=True):
        # History liss holds previous AND CURRENT Z
        self.Z_history.append(self.cur_Z.data.cpu().numpy())
        self.undoZ_button.setEnabled(len(self.Z_history)>1) # Enabling undo only when list is longer than 1, because the last item in the list is the current Z
        if clear_redo_list:
            self.Z_redo_list.clear()
            self.redoZ_button.setEnabled(False)

    def Undo_Z(self):
        self.Z_redo_list.append(self.Z_history.pop())
        self.cur_Z = torch.from_numpy(self.Z_history[-1]).type(self.cur_Z.dtype).to(self.cur_Z.device)
        self.ReProcess(dont_update_undo_list=True)
        self.undoZ_button.setEnabled(len(self.Z_history)>1) # Enabling undo only when list is longer than 1, because the last item in the list is the current Z
        self.redoZ_button.setEnabled(True)

    def Redo_Z(self):
        self.cur_Z = torch.from_numpy(self.Z_redo_list.pop()).type(self.cur_Z.dtype).to(self.cur_Z.device)
        self.ReProcess(dont_update_undo_list=True)
        self.redoZ_button.setEnabled(len(self.Z_redo_list)>0)
        self.Add_Z_2_history(clear_redo_list=False)

    def Z_undo_redo_enabling(self,enable):
        if enable:
            self.undoZ_button.setEnabled(len(self.Z_history)>1) # Enabling undo only when list is longer than 1, because the last item in the list is the current Z
            self.redoZ_button.setEnabled(len(self.Z_redo_list)>0)
        else:
            for button_name in ['undo','redo']:
                getattr(self, button_name + 'Z_button').setEnabled(enable)

    def save_file_and_Z_map(self):
        """
        Save active canvas and cur_Z map to image file.
        :return:
        """
        ALSO_SAVE_GRAYSCALE = self.JPEG_GUI and False
        self.saved_outputs_counter = 0
        while True:
            i,path = 0,[]
            while True:
                if self.canvas.opt['path']['results_root'].split('/')[i]=='results':
                    break
                path.append(self.canvas.opt['path']['results_root'].split('/')[i])
                i += 1
            path = os.path.join('/'.join(path),'GUI_outputs','%s_%d%s.png'%(self.image_name+('_QF%d'%(self.QF_box.value()) if self.JPEG_GUI else ''),self.saved_outputs_counter,'%s'))
            if not os.path.isfile(path%('')):
                break
            self.saved_outputs_counter += 1
        if path:
            imageio.imsave(path%(''),np.clip(255*self.canvas.output_image_0_1[0].data.cpu().numpy().transpose(1,2,0),0,255).astype(np.uint8))
            Z_image = np.clip(255/2/self.max_SVD_Lambda*(self.max_SVD_Lambda+self.cur_Z[0].data.cpu().numpy().transpose(1,2,0)),0,255).astype(np.uint8)
            if Z_image.shape[2] in [3,8**2]:
                if Z_image.shape[2]==8**2:
                    Z_image = util.Z_64channels2image(Z_image)
                imageio.imsave(path%('_Z'),Z_image)
            input_im_4_saving = self.input_image.squeeze(0).data.cpu().numpy().transpose((1, 2, 0))
            imageio.imsave(path.replace('_%d'%(self.saved_outputs_counter),'') % ('_Comp' if self.JPEG_GUI else '_LR'),input_im_4_saving)
            SAVE_JPEG = self.JPEG_GUI and False
            if SAVE_JPEG:
                input_im_4_saving = PIL_Image.fromarray((255*input_im_4_saving).astype(np.uint8))
                jpeg_path = path.replace('_%d'%(self.saved_outputs_counter),'').replace('.png','.jpg') % ('_Comp')
                # input_im_4_saving.save(jpeg_path,'jpeg',subsampling=2,qtables={0:list(self.Q_Tables[0].reshape([-1])),1:list(self.Q_Tables[1].reshape([-1]))})
                input_im_4_saving.save(jpeg_path,'jpeg',subsampling=2,qtables={0:list(np.clip(self.Q_Tables[0],0,255).reshape([-1])),1:list(np.clip(self.Q_Tables[1],0,255).reshape([-1]))})
                jpeg_MeanAbsErr = np.abs(np.array(PIL_Image.open(jpeg_path)).astype(np.float)-np.array(input_im_4_saving).astype(np.float)).mean()

            if ALSO_SAVE_GRAYSCALE:
                imageio.imsave(path % '_gray',rgb2ycbcr(self.canvas.output_image_0_1[0].data.cpu().numpy().transpose(1,2,0)))
                imageio.imsave(path % ('_gray_scribbled'), rgb2ycbcr(self.canvas.image_4_scribbling))
            if self.display_ESRGAN and self.saved_outputs_counter==0:
                imageio.imsave(path % ('_ESRGAN'), np.clip(255*self.ESRGAN_SR[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8))
            if self.canvas.current_display_index == self.canvas.scribble_display_index:
                self.Update_Scribble_Data()
            data_2_save = {}
            if ('current_scribble_mask' in self.canvas.__dict__.keys() and np.any(self.canvas.current_scribble_mask)):
                data_2_save['scribble_image'] = self.canvas.image_4_scribbling
                data_2_save['scribble_mask'] = self.canvas.current_scribble_mask
                imageio.imsave(path%('_scribbled'),self.canvas.image_4_scribbling)
            if self.using_estimated_kernel:
                data_2_save['estimated_kernel'] = self.estimated_kernel
            if len(data_2_save.keys())>0:
                np.savez(path.replace('.png','.npz')%('_scribble_data'),**data_2_save)
            print('Saved image %s'%(path%(''))+(' JPEG error: %.2f'%(jpeg_MeanAbsErr) if SAVE_JPEG else ''))
            self.statusBar.showMessage('Saved image %s'%(path%(''))+(' JPEG error: %.2f'%(jpeg_MeanAbsErr) if SAVE_JPEG else ''),INFO_MESSAGE_DURATION)
            self.saved_outputs_counter += 1

if __name__ == '__main__':

    app = QApplication([])
    window = MainWindow()
    app.exec_()