from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets,QtCore
import tkinter
from MainWindow import Ui_MainWindow
import scipy.io
import os
import random
import types

# EDitable SR imports:
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'KernelGAN'))
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
import matplotlib
from skimage.transform import resize
from scipy.signal import find_peaks
from DTE.imresize_DTE import imresize
import time
from collections import deque
from KernelGAN import train as KernelGAN
import copy

DISPLAY_ZOOM_FACTOR = 1
DISPLAY_ZOOM_FACTORS_RANGE = [1,4]
DOWNSCALED_HIST_VERSIONS = False#0.9
MIN_DOWNSCALING_4_HIST = 0.75
BRUSH_MULT = 3
SPRAY_PAINT_MULT = 5
SPRAY_PAINT_N = 100
USE_SVD = True
VERBOSITY = False
# MAX_SVD_LAMBDA = 1.
Z_OPTIMIZER_INITIAL_LR = 1e-1
Z_OPTIMIZER_INITIAL_LR_4_RANDOM = 1e-1
NUM_RANDOM_ZS = 1
VGG_RANDOM_DOMAIN = False
LOAD_HR_IMAGE = True
# DISPLAY_ESRGAN_RESULTS = True
# ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS = True
DISPLAY_INDUCED_LR = False
DICTIONARY_REPLACES_HISTOGRAM = True
L1_REPLACES_HISTOGRAM = False
NO_DC_IN_PATCH_HISTOGRAM = True
RELATIVE_STD_OPT = True
LOCAL_STD_4_OPT = True
ONLY_MODIFY_MASKED_AREA_WHEN_OPTIMIZING = False
D_EXPECTED_LR_SIZE = 64
ITERS_PER_OPT_ROUND = 5
HIGH_OPT_ITERS_LIMIT = True
MARGINS_AROUND_REGION_OF_INTEREST = 30
RANDOM_OPT_INITS = False
MULTIPLE_OPT_INITS = True
AUTO_CYCLE_LENGTH_4_PERIODICITY = True
Z_HISTORY_LENGTH = 10
LR_INTERPOLATION_4_SAVING = 'NN'

assert not (DICTIONARY_REPLACES_HISTOGRAM and L1_REPLACES_HISTOGRAM)

COLORS = [
    '#000000', '#82817f', '#820300', '#868417', '#007e03', '#037e7b', '#040079',
    '#81067a', '#7f7e45', '#05403c', '#0a7cf6', '#093c7e', '#7e07f9', '#7c4002',

    '#ffffff', '#c1c1c1', '#f70406', '#fffd00', '#08fb01', '#0bf8ee', '#0000fa',
    '#b92fc2', '#fffc91', '#00fd83', '#87f9f9', '#8481c4', '#dc137d', '#fb803c',
]

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]

SCRIBBLE_MODES = ['pencil','line', 'polygon','ellipse', 'rect','im_input','im_input_auto_location']
MODES = [
    'selectpoly', 'selectrect','indicatePeriodicity',
    #'eraser', 'fill',
    #'dropper', 'stamp',
    'dropper',
    #'spray', 'text',
    #'line', #'rand_Z',#'polyline',
    #'roundrect',
]+SCRIBBLE_MODES

LOCAL_TV_MASK_IDENTIFIERS_RANGE = [3,50]

IMPRINT_SIZE_CHANGES = ['narrower','wider','taller','shorter']
IMPRINT_LOCATION_CHANGES = ['left','right','up','down']

CANVAS_DIMENSIONS = 600, 400

# STAMP_DIR = './stamps'
# STAMPS = [os.path.join(STAMP_DIR, f) for f in os.listdir(STAMP_DIR)]

SELECTION_PEN = QPen(QColor(0xff, 0xff, 0xff), 1, Qt.DashLine)
PREVIEW_PEN = QPen(QColor(0xff, 0xff, 0xff), 1, Qt.SolidLine)


def build_font(config):
    """
    Construct a complete font from the configuration options
    :param self:
    :param config:
    :return: QFont
    """
    font = config['font']
    font.setPointSize(config['fontsize'])
    font.setBold(config['bold'])
    font.setItalic(config['italic'])
    font.setUnderline(config['underline'])
    return font


class Canvas(QLabel):

    mode = 'rectangle'
    secondary_color = QColor(Qt.white)

    primary_color_updated = pyqtSignal(str)
    secondary_color_updated = pyqtSignal(str)

    # Store configuration settings, including pen width, fonts etc.
    config = {
        # Drawing options.
        'size': 1,
        'fill': True,
        # Font options.
        'font': QFont('Times'),
        'fontsize': 12,
        'bold': False,
        'italic': False,
        'underline': False,
    }

    active_color = None
    preview_pen = None

    timer_event = None

    current_stamp = None

    def initialize(self):
        self.background_color = QColor(self.secondary_color) if self.secondary_color else QColor(Qt.white)
        self.eraser_color = QColor(self.secondary_color) if self.secondary_color else QColor(Qt.white)
        self.eraser_color.setAlpha(100)
        self.reset()

    def reset(self,canvas_dimensions=CANVAS_DIMENSIONS):
        # Create the pixmap for display.
        self.setPixmap(QPixmap(*canvas_dimensions))

        # Clear the canvas.
        self.pixmap().fill(self.background_color)

    def set_primary_color(self, hex):
        self.primary_color = QColor(hex)
        self.primary_button.setStyleSheet('QPushButton { background-color: %s; }' % hex)
        self.color_state = 0
        transparent_icon = QIcon()
        transparent_icon.addPixmap(QPixmap("icons/transparent.png"), QIcon.Normal, QIcon.Off)
        self.primary_button.setIcon(transparent_icon)

    def cycle_color_state(self):
        self.color_state = np.mod(self.color_state+1,4)
        if self.color_state==0:
            transparent_icon = QIcon()
            transparent_icon.addPixmap(QPixmap("icons/transparent.png"), QIcon.Normal, QIcon.Off)
            self.primary_button.setIcon(transparent_icon)
            # self.primary_button.setStyleSheet('QPush_button { background-color: %s; }' % self.primary_color.name())
        elif self.color_state==1:
            brightness_up_icon = QIcon()
            brightness_up_icon.addPixmap(QPixmap("icons/brightness_increase.png"), QIcon.Normal, QIcon.Off)
            self.primary_button.setIcon(brightness_up_icon)
        elif self.color_state == 2:
            brightness_down_icon = QIcon()
            brightness_down_icon.addPixmap(QPixmap("icons/brightness_decrease.png"), QIcon.Normal,QIcon.Off)
            self.primary_button.setIcon(brightness_down_icon)
        elif self.color_state == 3:
            brightness_down_icon = QIcon()
            brightness_down_icon.addPixmap(QPixmap("icons/fixed_color.png"), QIcon.Normal, QIcon.Off)
            self.primary_button.setIcon(brightness_down_icon)

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
                # self.local_TV_identifier += 1
                return QColor(np.mod(self.cyclic_color_shift,255), 255,np.mod(50*self.local_TV_identifier,255))

    def set_secondary_color(self, hex):
        self.secondary_color = QColor(hex)

    def set_config(self, key, value):
        self.config[key] = value

    def set_mode(self, mode):
        # Clean up active timer animations.
        self.timer_cleanup()
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

    def timer_cleanup(self):
        if self.timer_event:
            # Stop the timer, then trigger cleanup.
            timer_event = self.timer_event
            self.timer_event = None
            timer_event(final=True)

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
            self.Z_optimizer_Reset()
            self.SelectImage2Display(self.scribble_display_index)
            self.Add_scribble_2_Undo_list()
            self.imprinting_arrows_enabling(False)
            if self.color_state==3: # Advancing the local_TV_identifier cycle by 1:
                self.local_TV_identifier = np.mod(self.local_TV_identifier+1-LOCAL_TV_MASK_IDENTIFIERS_RANGE[0],np.diff(LOCAL_TV_MASK_IDENTIFIERS_RANGE)[0])+LOCAL_TV_MASK_IDENTIFIERS_RANGE[0]
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def any_scribbles_within_mask(self):
        return np.any(self.Z_mask_display_size*qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:, :, 0])

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            returnable =  fn(e)
            if self.mode in self.scribble_modes and not self.within_drawing:
                self.apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
                self.loop_apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
            return returnable

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            returnable =  fn(e)
            if self.mode in self.scribble_modes and not self.within_drawing:
                self.apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
                self.loop_apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
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

    # Mode-specific events.

    # Select polygon events

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
        # display_size_mask = self.in_picking_desired_hist_mode and self.desired_im_taken_from_same
        # self.HR_selected_mask = np.zeros([self.display_zoom_factor*v for v in self.HR_size]) if display_size_mask else np.zeros(self.HR_size)
        self.HR_selected_mask = np.zeros(self.HR_size)
        self.LR_mask_vertices = [(int(np.round(p.x()/self.DTE_opt['scale']/self.display_zoom_factor)),int(np.round(p.y()/self.DTE_opt['scale']/self.display_zoom_factor))) for p in (self.history_pos + [self.current_pos])]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        # I used to use HR mask that is pixel-algined in the LR domain, now changed to make sure it is pixel-aligned only in the HR domain (avoiding subpixel shifts due to self.display_zoom_factor):
        # self.HR_mask_vertices = [(coord[0]*self.DTE_opt['scale'],coord[1]*self.DTE_opt['scale']) for coord in self.LR_mask_vertices]
        self.HR_mask_vertices = [(int(np.round(p.x()/self.display_zoom_factor)),int(np.round(p.y()/self.display_zoom_factor))) for p in (self.history_pos + [self.current_pos])]
        # self.HR_mask_vertices_display_size = [(p.x(),p.y()) for p in (self.history_pos + [self.current_pos])]
        # self.HR_selected_mask = cv2.fillPoly(self.HR_selected_mask,[np.array(self.HR_mask_vertices_display_size if display_size_mask else self.HR_mask_vertices)],(1,1,1))
        self.HR_selected_mask = cv2.fillPoly(self.HR_selected_mask,[np.array(self.HR_mask_vertices)],(1,1,1))
        self.Z_mask = np.zeros(self.Z_size)
        if self.HR_Z:
            self.Z_mask = cv2.fillPoly(self.Z_mask, [np.array(self.HR_mask_vertices)], (1, 1, 1))
        else:
            self.Z_mask = cv2.fillPoly(self.Z_mask,[np.array(self.LR_mask_vertices)],(1,1,1))
        self.update_Z_mask_display_size()
        self.Update_Z_Sliders()
        self.Z_optimizer_Reset()
        self.selectpoly_button.setChecked(False)
        self.timer_cleanup()
        self.Avoid_Scribble_Display(False)
        if not self.in_picking_desired_hist_mode:
            self.apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
            self.loop_apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
        # self.selectpoly_copy()#I add this to remove the dashed selection lines from the image, after I didn't find any better way. This removes it if done immediatly after selection, for some yet to be known reason

    def update_Z_mask_display_size(self):
        self.Z_mask_display_size = \
            util.ResizeCategorialImage(self.Z_mask.astype(np.int16),dsize=tuple([self.display_zoom_factor*val for val in self.HR_size])).astype(self.Z_mask.dtype)

    def indicatePeriodicity_mousePressEvent(self, e):
        if not self.locked or e.button == Qt.RightButton:
            self.active_shape_fn = 'drawPolygon'
            self.preview_pen = SELECTION_PEN
            if self.history_pos is None:
                self.Avoid_Scribble_Display(True)
            self.generic_poly_mousePressEvent(e)
        if len(self.history_pos)==3:
            self.locked = True
            # self.generic_poly_mousePressEvent(e)
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
                    # return np.sqrt(sum([(points[0][axis]-points[1][axis])**2 for axis in range(2)]))

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
                self.periodicity_mag_1.setValue(np.linalg.norm(self.periodicity_points[0]))
                self.periodicity_mag_2.setValue(np.linalg.norm(self.periodicity_points[1]))
            else:
                self.periodicity_points = [(p.y()-self.history_pos[0].y(),p.x()-self.history_pos[0].x()) for p in self.history_pos[1:]]
            # self.timer_event(final=True)
            # self.reset_mode()
            SHOW_CHOSEN_POINTS = True
            # self.indicatePeriodicity_button.setDown(False)
            if SHOW_CHOSEN_POINTS:
                self.timer_cleanup()
                # self.active_shape_fn = None
                # self.periodicity_points_pen = QPen(QColor(0xff, 0x00, 0x00), 5, Qt.SolidLine)
                p = QPainter(self.pixmap())
                p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
                p.setPen(QPen(QColor(0xff, 0x00, 0x00), 5, Qt.SolidLine))
                # getattr(p, self.active_shape_fn)(*self.history_pos[:1] + [QPoint(np.round(point[1])+self.history_pos[0].x(),
                #                                                                  np.round(point[0])+self.history_pos[0].y()) for point in self.periodicity_points])
                getattr(p, 'drawPoint')(self.history_pos[0].x(),self.history_pos[0].y())
                for point in self.periodicity_points:
                    getattr(p, 'drawPoint')(np.round(point[1])+self.history_pos[0].x(), np.round(point[0])+self.history_pos[0].y())

            # self.Avoid_Scribble_Display(False)
    def indicatePeriodicity_timerEvent(self, final=False):
        self.generic_poly_timerEvent(final)

    def indicatePeriodicity_mouseMoveEvent(self, e):
        if not self.locked:
            self.generic_poly_mouseMoveEvent(e)

    def selectpoly_copy(self):
        """
        Copy a polygon region from the current image, returning it.

        Create a mask for the selected area, and use it to blank
        out non-selected regions. Then get the bounding rect of the
        selection and crop to produce the smallest possible image.

        :return: QPixmap of the copied region.
        """
        self.timer_cleanup()

        pixmap = self.pixmap().copy()
        bitmap = QBitmap(*CANVAS_DIMENSIONS)
        bitmap.clear()  # Starts with random data visible.

        p = QPainter(bitmap)
        # Construct a mask where the user selected area will be kept, the rest removed from the image is transparent.
        userpoly = QPolygon(self.history_pos + [self.current_pos])
        p.setPen(QPen(Qt.color1))
        p.setBrush(QBrush(Qt.color1))  # Solid color, Qt.color1 == bit on.
        p.drawPolygon(userpoly)
        p.end()

        # Set our created mask on the image.
        pixmap.setMask(bitmap)

        # Calculate the bounding rect and return a copy of that region.
        return pixmap.copy(userpoly.boundingRect())

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
        self.mask_bounding_rect = np.array(cv2.boundingRect(np.stack([list(p) for p in self.LR_mask_vertices], 1).transpose()))
        self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.mask_bounding_rect[2:]]))
        self.contained_Z_mask = True

    def selectrect_mouseReleaseEvent(self, e):
        self.current_pos = e.pos()
        self.locked = True
        # display_size_mask = self.in_picking_desired_hist_mode and self.desired_im_taken_from_same
        # self.HR_selected_mask = np.zeros([self.display_zoom_factor*v for v in self.HR_size]) if display_size_mask else np.zeros(self.HR_size)
        self.HR_selected_mask = np.zeros(self.HR_size)
        # self.HR_selected_mask = np.zeros(self.HR_size)
        self.LR_mask_vertices = [(int(np.round(p.x()/self.DTE_opt['scale']/self.display_zoom_factor)),int(np.round(p.y()/self.DTE_opt['scale']/self.display_zoom_factor))) for p in [self.origin_pos, self.current_pos]]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        # I used to use HR mask that is pixel-algined in the LR domain, now changed to make sure it is pixel-aligned only in the HR domain (avoiding subpixel shifts due to self.display_zoom_factor):
        # self.HR_mask_vertices = [(coord[0]*self.DTE_opt['scale'],coord[1]*self.DTE_opt['scale']) for coord in self.LR_mask_vertices]
        self.HR_mask_vertices = [(int(np.round(p.x()/self.display_zoom_factor)),int(np.round(p.y()/self.display_zoom_factor))) for p in [self.origin_pos, self.current_pos]]
        # self.HR_mask_vertices_display_size = [(p.x(),p.y()) for p in [self.origin_pos, self.current_pos]]
        # if display_size_mask:
        #     self.HR_selected_mask = cv2.rectangle(self.HR_selected_mask,self.HR_mask_vertices_display_size[0],self.HR_mask_vertices_display_size[1],(1,1,1),cv2.FILLED)
        # else:
        self.HR_selected_mask = cv2.rectangle(self.HR_selected_mask,self.HR_mask_vertices[0],self.HR_mask_vertices[1],(1,1,1),cv2.FILLED)
        self.Z_mask = np.zeros(self.Z_size)
        if self.HR_Z:
            self.Z_mask = cv2.rectangle(self.Z_mask, self.HR_mask_vertices[0], self.HR_mask_vertices[1], (1, 1, 1),cv2.FILLED)
        else:
            self.Z_mask = cv2.rectangle(self.Z_mask,self.LR_mask_vertices[0],self.LR_mask_vertices[1],(1,1,1),cv2.FILLED)
        self.update_Z_mask_display_size()
        self.Update_Z_Sliders()
        self.Z_optimizer_Reset()
        self.selectrect_button.setChecked(False)#This does not work, probably because of some genral property set for all "mode" buttons.
        self.timer_cleanup()
        self.Avoid_Scribble_Display(False)
        if not self.in_picking_desired_hist_mode:
            self.apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
            self.loop_apply_scribble_button.setEnabled(self.any_scribbles_within_mask())
        # self.selectrect_copy()  # I add this to remove the dashed selection lines from the image, after I didn't find any better way. This removes it if done immediatly after selection, for some yet to be known reason

    def Z_optimizer_Reset(self):
        self.Z_optimizer_initial_LR = Z_OPTIMIZER_INITIAL_LR
        self.Z_optimizer = None
        self.Z_optimizer_logger = None

    def ReturnMaskedMapAverage(self,map):
        return np.sum(map*self.Z_mask)/np.sum(self.Z_mask)

    def Update_Z_Sliders(self):
        sliderZ0_new_val = self.ReturnMaskedMapAverage(self.control_values[0].data.cpu().numpy())
        sliderZ1_new_val = self.ReturnMaskedMapAverage(self.control_values[1].data.cpu().numpy())
        slider_third_channel_new_val = self.ReturnMaskedMapAverage(self.control_values[2].data.cpu().numpy())
        self.sliderZ0.setSliderPosition(100*sliderZ0_new_val)
        self.sliderZ1.setSliderPosition(100*sliderZ1_new_val)
        self.slider_third_channel.setSliderPosition(100*slider_third_channel_new_val)
        self.previous_sliders_values = np.expand_dims(self.Z_mask,0)*np.array([sliderZ0_new_val,sliderZ1_new_val,slider_third_channel_new_val]).reshape([3,1,1])+\
                                       np.expand_dims(1-self.Z_mask, 0)*self.previous_sliders_values

    def selectrect_copy(self):
        """
        Copy a rectangle region of the current image, returning it.

        :return: QPixmap of the copied region.
        """
        self.timer_cleanup()
        return self.pixmap().copy(QRect(self.origin_pos, self.current_pos))

    # Eraser events

    def eraser_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def eraser_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.eraser_color, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())

            self.last_pos = e.pos()
            self.update()

    def eraser_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Stamp (pie) events

    def stamp_mousePressEvent(self, e):
        p = QPainter(self.pixmap())
        stamp = self.current_stamp
        p.drawPixmap(e.x() - stamp.width() // 2, e.y() - stamp.height() // 2, stamp)
        self.update()

    # Pen events

    def pen_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)
        self.pen_mouseMoveEvent(e)

    def pen_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), self.config['size']*self.display_zoom_factor, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            # print(self.last_pos, e.pos())
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

    def pen_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Brush events

    def brush_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def brush_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), self.config['size']*self.display_zoom_factor * BRUSH_MULT, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())

            self.last_pos = e.pos()
            self.update()

    def brush_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Spray events

    def spray_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def spray_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.Scribble_Color(), 1))

            for n in range(self.config['size']*self.display_zoom_factor * SPRAY_PAINT_N):
                xo = random.gauss(0, self.config['size']*self.display_zoom_factor * SPRAY_PAINT_MULT)
                yo = random.gauss(0, self.config['size']*self.display_zoom_factor * SPRAY_PAINT_MULT)
                p.drawPoint(e.x() + xo, e.y() + yo)

        self.update()

    def spray_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Text events

    def keyPressEvent(self, e):
        if self.mode == 'text':
            if e.key() == Qt.Key_Backspace:
                self.current_text = self.current_text[:-1]
            else:
                self.current_text = self.current_text + e.text()

    def text_mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.current_pos is None:
            self.current_pos = e.pos()
            self.current_text = ""
            self.timer_event = self.text_timerEvent

        elif e.button() == Qt.LeftButton:

            self.timer_cleanup()
            # Draw the text to the image
            p = QPainter(self.pixmap())
            p.setRenderHints(QPainter.Antialiasing)
            font = build_font(self.config)
            p.setFont(font)
            pen = QPen(self.primary_color, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            p.drawText(self.current_pos, self.current_text)
            self.update()

            self.reset_mode()

        elif e.button() == Qt.RightButton and self.current_pos:
            self.reset_mode()

    def text_timerEvent(self, final=False):
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = PREVIEW_PEN
        p.setPen(pen)
        if self.last_text:
            font = build_font(self.last_config)
            p.setFont(font)
            p.drawText(self.current_pos, self.last_text)

        if not final:
            font = build_font(self.config)
            p.setFont(font)
            p.drawText(self.current_pos, self.current_text)

        self.last_text = self.current_text
        self.last_config = self.config.copy()
        self.update()

    # Fill events

    def fill_mousePressEvent(self, e):

        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color

        image = self.pixmap().toImage()
        w, h = image.width(), image.height()
        x, y = e.x(), e.y()

        # Get our target color from origin.
        target_color = image.pixel(x,y)

        have_seen = set()
        queue = [(x, y)]

        def get_cardinal_points(have_seen, center_pos):
            points = []
            cx, cy = center_pos
            for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                xx, yy = cx + x, cy + y
                if (xx >= 0 and xx < w and
                    yy >= 0 and yy < h and
                    (xx, yy) not in have_seen):

                    points.append((xx, yy))
                    have_seen.add((xx, yy))

            return points

        # Now perform the search and fill.
        p = QPainter(self.pixmap())
        p.setPen(QPen(self.active_color))

        while queue:
            x, y = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QPoint(x, y))
                queue.extend(get_cardinal_points(have_seen, (x, y)))

        self.update()

    # Dropper events

    def dropper_mousePressEvent(self, e):
        c = self.pixmap().toImage().pixel(e.pos())
        hex = QColor(c).name()

        if e.button() == Qt.LeftButton:
            self.set_primary_color(hex)
            self.primary_color_updated.emit(hex)  # Update UI.

        elif e.button() == Qt.RightButton:
            self.set_secondary_color(hex)
            self.secondary_color_updated.emit(hex)  # Update UI.

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
        if self.last_pos:
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.last_pos), *self.active_shape_args)

        if not final:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.current_pos), *self.active_shape_args)
        # else:
        #     print('Now its final')
            # print(self.current_pos)
        # self.dash_offset = 0

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
                # p.setBrush(QBrush(self.secondary_color))
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
            getattr(scribble_mask, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
            self.scribble_mask_canvas.update()
            self.update()

        self.reset_mode()

    # Line events

    def line_mousePressEvent(self, e):
        self.origin_pos = e.pos()
        self.current_pos = e.pos()
        self.preview_pen = PREVIEW_PEN
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
        # self.timer_cleanup()
        # self.active_shape_fn = None
        periodicity_points_pen = QPen(QColor(0xff, 0x00, 0x00), 5, Qt.SolidLine)
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        p.setPen(periodicity_points_pen)
        # getattr(p, self.active_shape_fn)(*self.history_pos[:1] + [QPoint(np.round(point[1])+self.history_pos[0].x(),
        #                                                                  np.round(point[0])+self.history_pos[0].y()) for point in self.periodicity_points])
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

    # Polyline events

    def polyline_mousePressEvent(self, e):
        self.active_shape_fn = 'drawPolyline'
        self.preview_pen = PREVIEW_PEN
        self.generic_poly_mousePressEvent(e)

    def polyline_timerEvent(self, final=False):
        self.generic_poly_timerEvent(final)

    def polyline_mouseMoveEvent(self, e):
        self.generic_poly_mouseMoveEvent(e)

    def polyline_mouseDoubleClickEvent(self, e):
        self.generic_poly_mouseDoubleClickEvent(e)

    # Rectangle events

    def rect_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.active_shape_args = ()
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def rect_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def rect_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def rect_mouseReleaseEvent(self, e):
        self.generic_shape_mouseReleaseEvent(e)

    # Input image scribble events:

    def im_input_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.active_shape_args = ()
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def im_input_auto_location_mousePressEvent(self, e):
        self.im_input_mousePressEvent(e)

    def im_input_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def im_input_auto_location_timerEvent(self, final=False):
        self.im_input_timerEvent(final)

    def im_input_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def im_input_auto_location_mouseMoveEvent(self, e):
        self.im_input_mouseMoveEvent(e)

    def im_input_mouseReleaseEvent(self, e):
        self.imprinting_location_boundaries = None
        self.finalize_im_input(e,transparent_mask=self.special_behavior_button.isChecked())

    def im_input_auto_location_mouseReleaseEvent(self, e):
        if self.imprinting_location_boundaries is None:
            self.imprinting_location_boundaries = []
        self.finalize_im_input(e,transparent_mask=self.special_behavior_button.isChecked())
    def FindOptimalImprintingLocation(self,desired_mask_bounding_rect):
        NUM_BEST_2_KEEP = 4
        NUM_SAMPLES_IN_RANGE = 10*NUM_BEST_2_KEEP
        def crop_LR_im(cropping_location):
            return self.SR_model.model_input.data[0].cpu().numpy().transpose(1,2,0)[cropping_location[0]:cropping_location[2],cropping_location[1]:cropping_location[3]:,-3:]
        HR_im_projected_2_ortho_nullspace = self.Project_2_Orthog_Nullspace(self.SR_model.fake_H.data[0].cpu().numpy().transpose(1,2,0))
        def crop_HR_im(cropping_location):
            return HR_im_projected_2_ortho_nullspace[cropping_location[0]:cropping_location[2],cropping_location[1]:cropping_location[3]:,:]
        def return_average_abs_im_diff(existing_im_loc,desired_HR_im,desired_HR_im_mask,LR_phase):
            if LR_phase:
                existing_im = crop_LR_im(existing_im_loc)
                org_size = existing_im.shape[:2]
            else:
                org_size = np.array([existing_im_loc[2]-existing_im_loc[0],existing_im_loc[3]-existing_im_loc[1]])
                existing_im = crop_HR_im(existing_im_loc)
            resized_desired_im = util.ResizeScribbleImage(desired_HR_im,dsize=tuple([v*(self.DTE_opt['scale'] if LR_phase else 1) for v in org_size]))
            resized_desired_im_mask = util.ResizeCategorialImage(desired_HR_im_mask.astype(np.uint8),dsize=tuple([v*(self.DTE_opt['scale'] if LR_phase else 1) for v in org_size]))
            if LR_phase:
                resized_desired_im = imresize(resized_desired_im,1/self.DTE_opt['scale'])
                resized_desired_im_mask = imresize(resized_desired_im_mask,1/self.DTE_opt['scale'])!=0
            return np.sum(np.abs(resized_desired_im-existing_im)*np.expand_dims(resized_desired_im_mask,-1))/np.sum(resized_desired_im_mask)/3

        desired_image = self.Project_2_Orthog_Nullspace(self.desired_image[0])[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1] + desired_mask_bounding_rect[3],
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
                            samples_2_add[invalid_samples,dim_num] = np.random.randint(low=int(np.floor(dim_range[0] / (self.DTE_opt['scale'] if LR_phase else 1))),
                                high=1 + int(np.ceil(dim_range[1] / (self.DTE_opt['scale'] if LR_phase else 1))),size=[np.sum(invalid_samples)])
                            if dim_num>1:
                                invalid_samples = samples_2_add[:,dim_num]-samples_2_add[:,dim_num-2]<(1 if LR_phase else self.DTE_opt['scale'])
                            else:
                                invalid_samples = np.zeros_like(invalid_samples)

                    sampled_locations.append(samples_2_add.astype(np.int32))
                sampled_locations = np.concatenate(sampled_locations,0)
                average_LR_diffs = np.array([return_average_abs_im_diff(loc,desired_image,desired_image_mask,LR_phase=LR_phase) for loc in sampled_locations])
                leading_location_inds = np.argsort(average_LR_diffs)
                if average_LR_diffs[leading_location_inds[0]]>=latest_LR_diff:
                    if LR_phase:
                        half_range = int(np.ceil(self.DTE_opt['scale'] / 2))
                        self.imprinting_location_boundaries = [[keep_within_range(best_locations[0]*self.DTE_opt['scale'] + v) for v in [-half_range, half_range]]]
                        samples_per_best_location = [NUM_SAMPLES_IN_RANGE]
                    break
                latest_LR_diff = average_LR_diffs[leading_location_inds[0]]
                best_locations = sampled_locations[leading_location_inds[:NUM_BEST_2_KEEP]]
                self.imprinting_location_boundaries = [[keep_within_range((self.DTE_opt['scale'] if LR_phase else 1)*(best_location + v)) for v in [-1, 1]] for best_location in best_locations]
                # Distributing number of samples per chosen location according to the locations' scores:
                samples_per_best_location = 1/np.maximum(0.01,(average_LR_diffs[leading_location_inds[:NUM_BEST_2_KEEP]]))
                samples_per_best_location = (samples_per_best_location/np.sum(samples_per_best_location)*NUM_SAMPLES_IN_RANGE).astype(int)
                samples_per_best_location[-1] = NUM_SAMPLES_IN_RANGE-np.sum(samples_per_best_location[:-1])

        best_location = best_locations[0]
        self.target_imprinting_dimensions = np.array([np.abs(best_location[2] - best_location[0]) + 1, np.abs(best_location[3] - best_location[1]) + 1])
        self.top_left_corner = np.array([np.minimum(best_location[2], best_location[0]), np.minimum(best_location[3], best_location[1])])

    def finalize_im_input(self, e=None,transparent_mask=False,modification=None):
        EXPLORE_SHIFTS = False
        SIZE_MODIFICATION_STEP_SIZE = 1*self.display_zoom_factor  # if ALLOW_LR_SUBPIXEL_TARGET_SIZES else self.DTE_opt['scale']*self.display_zoom_factor
        # return
        explore_shifts = EXPLORE_SHIFTS and modification is None
        desired_mask_bounding_rect = np.array(cv2.boundingRect(np.stack([list(p) for p in self.desired_image_HR_mask_vertices], 1).transpose()))
        if self.last_pos and e is not None:
            # Clear up indicator.
            self.timer_cleanup()
            e_y,e_x,origin_y,origin_x = e.pos().y(),e.pos().x(),self.origin_pos.y(),self.origin_pos.x()
            def convert_2_saved_im_size(coord):
                return int(np.round(coord/self.display_zoom_factor))
            e_y, e_x,origin_y,origin_x = convert_2_saved_im_size(e_y),convert_2_saved_im_size(e_x),convert_2_saved_im_size(origin_y),convert_2_saved_im_size(origin_x)
            self.target_imprinting_dimensions = np.array([np.abs(e_y-origin_y)+1,np.abs(e_x-origin_x)+1])
            self.top_left_corner = np.array([np.minimum(e_y,origin_y),np.minimum(e_x,origin_x)])
        if modification is None:
            if not (self.last_pos and e is not None) or np.any(self.target_imprinting_dimensions<self.DTE_opt['scale']): #target dimensions are smaller than 2 pixels in the LR image::
                if self.desired_im_taken_from_same and self.imprinting_location_boundaries is None:  # If the desired image was taken from the one being edited, this means we want to place the desired image at its original location.
                    self.top_left_corner = desired_mask_bounding_rect[:2][::-1]
                    self.target_imprinting_dimensions = desired_mask_bounding_rect[2:][::-1]
                else:  # Otherwise, this was pressed by mistake.
                    print('Attempting to imprint into a too small region. Please try again.')
                    self.reset_mode()
                    return
            self.desired_im_HR_mask_4_imprinting = 1 * self.desired_image_HR_mask[0]
            if transparent_mask:
                GRAYLEVELS_TOLERANCE = 2
                transparency_mask = (np.sqrt(np.mean((np.round(255*self.desired_image[0])-self.primary_color.getRgb()[:3])**2,-1))<=GRAYLEVELS_TOLERANCE).astype(np.uint8)
                transparency_mask = cv2.morphologyEx(transparency_mask, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                def mask_negation(mask):
                    return (-1 * (mask.astype(np.float32) - 1)).astype(mask.dtype)
                self.desired_im_HR_mask_4_imprinting = np.logical_and(self.desired_im_HR_mask_4_imprinting,mask_negation(transparency_mask)).astype(self.desired_im_HR_mask_4_imprinting.dtype)
                # self.desired_im_HR_mask_4_imprinting = np.logical_and(self.desired_im_HR_mask_4_imprinting,np.expand_dims(mask_negation(transparency_mask),-1)).astype(self.desired_im_HR_mask_4_imprinting.dtype)
            # else:
            #     self.transparent_mask_bg_color = None
            if self.imprinting_location_boundaries is not None:
                self.imprinting_location_boundaries.append((origin_y, origin_x, e_y, e_x))
                if len(self.imprinting_location_boundaries) == 1:
                    print('Registered outer imprinting location bounds')
                    self.Undo_scribble(add_2_redo_list=False) #Remove the scribble that was just unnecessarily added to the undo list
                    self.reset_mode()
                    return
                else:
                    self.FindOptimalImprintingLocation(desired_mask_bounding_rect)
                    print('Automatically determined optimal imprinting location')
                    self.imprinting_location_boundaries = None
            self.cur_loc_step_size = SIZE_MODIFICATION_STEP_SIZE // (2 * self.display_zoom_factor)
        if modification is not None:
            self.SelectImage2Display(self.scribble_display_index)
            self.Undo_scribble(add_2_redo_list=False)
            self.Add_scribble_2_Undo_list()
            if modification in IMPRINT_LOCATION_CHANGES:
                if modification=='right':
                    self.top_left_corner[1] += self.display_zoom_factor
                elif modification=='left':
                    self.top_left_corner[1] -= self.display_zoom_factor
                elif modification=='up':
                    self.top_left_corner[0] -= self.display_zoom_factor
                elif modification=='down':
                    self.top_left_corner[0] += self.display_zoom_factor
            elif modification in IMPRINT_SIZE_CHANGES:
                if modification=='narrower':
                    self.target_imprinting_dimensions[1] -= SIZE_MODIFICATION_STEP_SIZE
                    self.top_left_corner[1] += self.cur_loc_step_size
                elif modification=='wider':
                    self.target_imprinting_dimensions[1] += SIZE_MODIFICATION_STEP_SIZE
                    self.top_left_corner[1] -= self.cur_loc_step_size
                elif modification=='shorter':
                    self.target_imprinting_dimensions[0] -= SIZE_MODIFICATION_STEP_SIZE
                    self.top_left_corner[0] += self.cur_loc_step_size
                elif modification=='taller':
                    self.target_imprinting_dimensions[0] += SIZE_MODIFICATION_STEP_SIZE
                    self.top_left_corner[0] -= self.cur_loc_step_size
                self.cur_loc_step_size = SIZE_MODIFICATION_STEP_SIZE-self.cur_loc_step_size
        # Extending the target region size to be an integer muliplication of the SR scale factor, otherwise it cannot correspond to any upscale of a low-res region:
        # extended_dimensions = (np.ceil(self.target_imprinting_dimensions/self.DTE_opt['scale']/self.display_zoom_factor)*self.DTE_opt['scale']*self.display_zoom_factor).astype(np.uint32)
        extended_dimensions = (np.ceil(self.target_imprinting_dimensions/self.DTE_opt['scale'])*self.DTE_opt['scale']).astype(np.uint32)
        target_im_pad_sizes = (extended_dimensions - self.target_imprinting_dimensions) // 2
        top_left_corner = np.maximum([0, 0], self.top_left_corner -target_im_pad_sizes)
        target_im_pad_sizes = np.stack([target_im_pad_sizes,extended_dimensions-self.target_imprinting_dimensions-target_im_pad_sizes],-1)
        def crop_target_im_using_selected_rectangle(array):
            return array[top_left_corner[0]:top_left_corner[0]+extended_dimensions[0],top_left_corner[1]:top_left_corner[1]+extended_dimensions[1],...]
        relevant_existing_scribble_image = 255*crop_target_im_using_selected_rectangle(self.SR_model.fake_H[0].data.cpu().numpy().transpose(1, 2, 0))

        def crop_desired_im_using_bounding_rect(array):
            return array[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1]+desired_mask_bounding_rect[3],
                                desired_mask_bounding_rect[0]:desired_mask_bounding_rect[0]+desired_mask_bounding_rect[2],...]
        IGNORE_DESIRED_MASK_4_COMBINATION = True
        cropped_desired_image = crop_desired_im_using_bounding_rect(self.desired_image[0])
        # self.desired_im_HR_mask_4_imprinting = 1 * self.desired_image_HR_mask[0]
        # if self.transparent_mask_bg_color is not None:
        #     transparency_mask = np.all(self.desired_image[0]==self.transparent_mask_bg_color,-1).astype(np.uint8)
        #     transparency_mask = cv2.morphologyEx(transparency_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        #     def mask_negation(mask):
        #         return (-1*(mask.astype(np.float32)-1)).astype(mask.dtype)
        #     self.desired_im_HR_mask_4_imprinting = np.logical_and(self.desired_im_HR_mask_4_imprinting,np.expand_dims(mask_negation(transparency_mask),-1)).astype(self.desired_im_HR_mask_4_imprinting.dtype)
        cropped_desired_image_mask = crop_desired_im_using_bounding_rect(self.desired_im_HR_mask_4_imprinting)
        rescaled_cropped_desired_image = 255*util.ResizeScribbleImage(cropped_desired_image,dsize=tuple(self.target_imprinting_dimensions))
        rescaled_cropped_desired_image_mask = np.expand_dims(util.ResizeCategorialImage(cropped_desired_image_mask.astype(np.uint8),dsize=tuple(self.target_imprinting_dimensions)),-1)
        # Padding desired regions to fit the size of relevant_existing_scribble_image, which is an integer multiplication of scale_facotr*display_factor:
        def zero_pad_desired_array(array,mode='constant'):
            return np.pad(array,(tuple(target_im_pad_sizes[0]),tuple(target_im_pad_sizes[1]),(0,0)),mode=mode)
        rescaled_cropped_desired_image = zero_pad_desired_array(rescaled_cropped_desired_image,mode='edge' if IGNORE_DESIRED_MASK_4_COMBINATION else 'constant')
        rescaled_cropped_desired_image_mask = zero_pad_desired_array(rescaled_cropped_desired_image_mask)
        # if self.transparent_mask_bg_color is not None:
        #     transparency_mask = np.all(rescaled_cropped_desired_image==self.transparent_mask_bg_color,-1).astype(np.uint8)
        #     transparency_mask = cv2.morphologyEx(transparency_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        #     def mask_negation(mask):
        #         return (-1*(mask.astype(np.float32)-1)).astype(mask.dtype)
        #     rescaled_cropped_desired_image_mask = np.logical_and(rescaled_cropped_desired_image_mask,np.expand_dims(mask_negation(transparency_mask),-1)).astype(rescaled_cropped_desired_image_mask.dtype)

        if not IGNORE_DESIRED_MASK_4_COMBINATION:
            rescaled_cropped_desired_image = rescaled_cropped_desired_image*rescaled_cropped_desired_image_mask+relevant_existing_scribble_image*(1-rescaled_cropped_desired_image_mask)
        combined_image_2_input = np.clip(255*util.ResizeScribbleImage(self.Enforce_DT_on_Image_Pair(LR_source=relevant_existing_scribble_image/255,
            HR_input=rescaled_cropped_desired_image/255),dsize=tuple(relevant_existing_scribble_image.shape[:2])),0,255)
        if explore_shifts:# Does not happend when called with modification:
            # At first I chose the combined image that had the biggest difference from the existing image, thinking it means it makes a lot of difference. But then I figured
            # I actually want the opposite, because all the difference the desired image can make is in the higher frequencies. So if anything, I want the combined image to look similar
            # to the current SR, assuming that the fact it exists implies it sits on the natural images manifold.
            most_influencial_location_index = np.argmin([np.sum((relevant_existing_scribble_image[i]-combined_image_2_input[i])**2) for i in range(len(relevant_existing_scribble_image))])
            combined_image_2_input = combined_image_2_input[most_influencial_location_index]
            self.top_left_corner = [self.top_left_corner[most_influencial_location_index]]
        INTEGRATING_SCRIBBLE_IN_MODEL_SIZE = False
        def integrate_patch_into_image(patch,image):
            f = 1 if INTEGRATING_SCRIBBLE_IN_MODEL_SIZE else self.display_zoom_factor
            image[f*top_left_corner[0]:f*top_left_corner[0]+f*extended_dimensions[0],f*top_left_corner[1]:f*top_left_corner[1]+f*extended_dimensions[1],...] = patch
            return image
        if INTEGRATING_SCRIBBLE_IN_MODEL_SIZE:
            new_scribble_image = integrate_patch_into_image(combined_image_2_input,255*self.SR_model.fake_H[0].data.cpu().numpy().transpose(1, 2, 0))
            new_scribble_image = util.ResizeScribbleImage(new_scribble_image,dsize=tuple([s*self.display_zoom_factor for s in new_scribble_image.shape[:2]]))
        else:
            combined_image_2_input = util.ResizeScribbleImage(combined_image_2_input,dsize=tuple([s*self.display_zoom_factor for s in combined_image_2_input.shape[:2]]))
            new_scribble_image = integrate_patch_into_image(combined_image_2_input,qimage2ndarray.rgb_view(self.pixmap().toImage()))
        self.setPixmap(QPixmap(qimage2ndarray.array2qimage(new_scribble_image)))

        # Taking care of scribble mask:
        existing_scribble_mask = qimage2ndarray.rgb_view(self.scribble_mask_canvas.pixmap().toImage())[:,:,0]
        if not INTEGRATING_SCRIBBLE_IN_MODEL_SIZE:
            model_size_scribble_mask = np.expand_dims(util.ResizeCategorialImage(existing_scribble_mask,dsize=tuple([s//self.display_zoom_factor for s in existing_scribble_mask.shape[:2]])),-1)
        existing_scribble_mask = np.expand_dims(existing_scribble_mask,-1)
        relevant_existing_scribble_image_mask = crop_target_im_using_selected_rectangle(model_size_scribble_mask)
        combined_image_mask_2_input = rescaled_cropped_desired_image_mask+(1-rescaled_cropped_desired_image_mask)*relevant_existing_scribble_image_mask
        if INTEGRATING_SCRIBBLE_IN_MODEL_SIZE:
            new_scribble_image_mask = integrate_patch_into_image(combined_image_mask_2_input,existing_scribble_mask)
            new_scribble_image_mask = util.ResizeCategorialImage(new_scribble_image_mask[:,:,0],dsize=tuple([s*self.display_zoom_factor for s in new_scribble_image_mask.shape[:2]]))
        else:
            combined_image_mask_2_input = util.ResizeCategorialImage(combined_image_mask_2_input[:,:,0],dsize=tuple([s*self.display_zoom_factor for s in combined_image_mask_2_input.shape[:2]]))
            new_scribble_image_mask = integrate_patch_into_image(combined_image_mask_2_input, existing_scribble_mask[:,:,0])
        self.scribble_mask_canvas.setPixmap(QPixmap(qimage2ndarray.array2qimage(np.repeat(np.expand_dims(new_scribble_image_mask,-1),3,-1))))
        self.imprinting_arrows_enabling(True)
        # for button_name in IMPRINT_LOCATION_CHANGES+IMPRINT_SIZE_CHANGES:
        #     getattr(self, button_name + '_imprinting_button').setEnabled(True)
        self.reset_mode()

    def imprinting_arrows_enabling(self,enable):
        for button_name in IMPRINT_LOCATION_CHANGES+IMPRINT_SIZE_CHANGES:
            getattr(self, button_name + '_imprinting_button').setEnabled(enable)

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
        self.preview_pen = PREVIEW_PEN
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
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def ellipse_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def ellipse_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def ellipse_mouseReleaseEvent(self, e):
        self.generic_shape_mouseReleaseEvent(e)

    # Roundedrect events

    def roundrect_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRoundedRect'
        self.active_shape_args = (25, 25)
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def roundrect_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def roundrect_mouseMoveEvent(self, e):
        self.generic_shape_mouseMoveEvent(e)

    def roundrect_mouseReleaseEvent(self, e):
        self.generic_shape_mouseReleaseEvent(e)


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        available_GPUs = util.Assign_GPU()
        self.num_random_Zs = NUM_RANDOM_ZS
        self.canvas = Canvas()
        self.setupUi(self)

        # Editable SR:
        self.opt = option.parse('./options/test/GUI_esrgan.json', is_train=False)
        self.opt = option.dict_to_nonedict(self.opt)
        matplotlib.use('Qt5Agg')
        matplotlib.interactive(True)
        self.auto_set_hist_temperature = False
        self.latest_optimizer_objective = ''
        # Replace canvas placeholder from QtDesigner.
        self.horizontalLayout.removeWidget(self.canvas)
        # self.canvas = Canvas()
        self.canvas.display_zoom_factor = 1*DISPLAY_ZOOM_FACTOR
        self.canvas.desired_image = None
        self.canvas.imprinting_location_boundaries = None
        self.canvas.Z_optimizer_Reset()
        self.canvas.DTE_opt = self.opt
        self.canvas.initialize()
        self.canvas.HR_Z = 'HR' in self.canvas.DTE_opt['network_G']['latent_input_domain']
        self.Initialize_SR_model(reprocess=False)
        self.using_estimated_kernel = False
        # We need to enable mouse tracking to follow the mouse without the button pressed.
        self.canvas.setMouseTracking(True)
        # Enable focus to capture key inputs.
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        # if not ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
        #     if LOAD_HR_IMAGE:
        #         #Add a 2nd canvas:
        #         self.GT_canvas = Canvas()
        #         self.GT_canvas.initialize()
        #         self.horizontalLayout.addWidget(self.GT_canvas)
        #     if self.display_ESRGAN:
        #         self.ESRGAN_canvas = Canvas()
        #         self.ESRGAN_canvas.initialize()
        #         self.horizontalLayout.addWidget(self.ESRGAN_canvas)

        if DISPLAY_INDUCED_LR:
            #Add a 3rd canvas:
            self.LR_canvas = Canvas()
            self.LR_canvas.initialize()
            self.horizontalLayout.addWidget(self.LR_canvas)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        for mode in MODES:
            btn = getattr(self, '%s_button' % mode)
            btn.pressed.connect(lambda mode=mode: self.canvas.set_mode(mode))
            mode_group.addButton(btn)

        # # Setup up action signals
        # self.actionCopy.triggered.connect(self.copy_to_clipboard)

        # Initialize animation timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.on_timer)
        self.timer.setInterval(100)
        self.timer.start()

        # Menu options
        self.open_HR_image_button.clicked.connect(lambda x: self.open_file(HR_image=True))
        self.open_image_button.clicked.connect(lambda x: self.open_file(HR_image=False))
        self.Z_load_button.pressed.connect(self.Load_Z)
        self.Z_mask_load_button.pressed.connect(self.Load_Z_mask)
        self.estimatedKenrel_button.pressed.connect(self.Change_kernel_in_use)


        self.ProcessRandZ_button.clicked.connect(lambda x: self.Process_Random_Z(limited=False))
        self.scribble_reset_button.pressed.connect(self.Reset_Image_4_Scribbling)
        self.apply_scribble_button.pressed.connect(lambda x:self.Optimize_Z('scribble'))
        self.loop_apply_scribble_button.pressed.connect(lambda x:self.Optimize_Z('scribble',loop=True))
        self.ProcessLimitedRandZ_button.clicked.connect(lambda x: self.Process_Random_Z(limited=True))

        # self.DisplayedImageSelectionButton.highlighted.connect(self.SelectImage2Display)
        # if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
        #     if DISPLAY_ESRGAN_RESULTS:
        #         self.DisplayedImageSelectionButton.addItem('ESRGAN')
        #         self.DisplayedImageSelectionButton.setEnabled(True)
        #         self.ESRGAN_index = self.DisplayedImageSelectionButton.findText('ESRGAN')
        #     if True: #LOAD_HR_IMAGE: Now I always add this display, and only enable it for images with GT
        #         self.DisplayedImageSelectionButton.addItem('GT')
        #         self.DisplayedImageSelectionButton.setEnabled(True)
        #         self.GT_HR_index = self.DisplayedImageSelectionButton.findText('GT')
        #     else:
        #         self.GT_HR_index = None
        # self.DisplayedImageSelectionButton.addItem('Z')
        # self.cur_Z_im_index = self.DisplayedImageSelectionButton.findText('Z')
        # self.canvas.cur_Z_im_index = self.cur_Z_im_index
        # self.canvas.current_display_index = 1*self.cur_Z_im_index
        # self.DisplayedImageSelectionButton.addItems([str(i+1) for i in range(self.num_random_Zs)])
        # self.random_display_indexes = [self.DisplayedImageSelectionButton.findText(str(i+1)) for i in range(self.num_random_Zs)]
        # self.DisplayedImageSelectionButton.addItem('Scribble')
        # self.canvas.scribble_display_index = self.DisplayedImageSelectionButton.findText('Scribble')

        self.CopyFromRandom_button.pressed.connect(self.CopyRandom2Default)
        self.Copy2Random_button.pressed.connect(self.CopyDefault2Random)
        self.IncreaseSTD_button.clicked.connect(lambda x:self.Optimize_Z('STD_increase' if RELATIVE_STD_OPT else 'max_STD'))
        self.DecreaseSTD_button.clicked.connect(lambda x:self.Optimize_Z('STD_decrease' if RELATIVE_STD_OPT else 'min_STD'))
        self.DecreaseTV_button.clicked.connect(lambda x:self.Optimize_Z('TV'))
        self.ImitateHist_button.clicked.connect(lambda x:self.Optimize_Z('hist'))
        self.ImitatePatchHist_button.clicked.connect(lambda x:self.Optimize_Z('patchhist'))
        self.FoolAdversary_button.clicked.connect(lambda x:self.Optimize_Z('Adversarial'))
        self.IncreasePeriodicity_2D_button.clicked.connect(lambda x:self.Optimize_Z('periodicity'))
        self.IncreasePeriodicity_1D_button.clicked.connect(lambda x:self.Optimize_Z('periodicity_1D'))
        # self.MatchSliders_button.clicked.connect(lambda x:self.Optimize_Z('desired_SVD'))

        self.unselect_button.clicked.connect(self.Clear_Z_Mask)
        self.invertSelection_button.clicked.connect(self.Invert_Z_Mask)
        self.uniformZ_button.clicked.connect(self.ApplyUniformZ)
        # self.special_behavior_button.clicked.connect(self.canvas.Z_optimizer_Reset)
        self.desiredAppearanceMode_button.clicked.connect(lambda checked: self.DesiredAppearanceMode(checked,another_image=False))
        self.Zdisplay_button.clicked.connect(self.ToggleDisplay_Z_Image)
        self.undoZ_button.clicked.connect(self.Undo_Z)
        self.canvas.undo_scribble_button = self.undo_scribble_button
        self.canvas.undo_scribble_button.clicked.connect(lambda s:self.canvas.Undo_scribble(add_2_redo_list=True))
        self.canvas.redo_scribble_button = self.redo_scribble_button
        self.canvas.redo_scribble_button.clicked.connect(self.canvas.Redo_scribble)
        self.redoZ_button.clicked.connect(self.Redo_Z)
        self.desiredExternalAppearanceMode_button.clicked.connect(lambda checked: self.DesiredAppearanceMode(checked,another_image=True))
        self.canvas.in_picking_desired_hist_mode = False
        if self.auto_hist_temperature_mode_Enabled:
            self.auto_hist_temperature_mode_button.clicked.connect(lambda checked:self.AutoHistTemperatureMode(checked))

        self.SaveImage_button.clicked.connect(self.save_file)
        self.SaveImageAndData_button.clicked.connect(self.save_file_and_Z_map)
        self.DecreaseDisplayZoom.setEnabled(self.canvas.display_zoom_factor>DISPLAY_ZOOM_FACTORS_RANGE[0])
        self.DecreaseDisplayZoom_button.clicked.connect(lambda x: self.Change_Display_Zoom(increase=False))
        self.IncreaseDisplayZoom.setEnabled(self.canvas.display_zoom_factor <DISPLAY_ZOOM_FACTORS_RANGE[1])
        self.IncreaseDisplayZoom_button.clicked.connect(lambda x: self.Change_Display_Zoom(increase=True))
        # self.ClearImage_button.clicked.connect(self.canvas.reset)
        self.InvertColors_button.clicked.connect(self.invert)
        self.FlipHorizontal_button.clicked.connect(self.flip_horizontal)
        self.FlipVertical_button.clicked.connect(self.flip_vertical)

        # sizeicon = QLabel()
        # sizeicon.setPixmap(QPixmap(os.path.join('images', 'border-weight.png')))
        # self.drawingToolbar.addWidget(sizeicon)
        # self.sizeselect = QSlider()
        # self.sizeselect.setRange(1,20)
        # self.sizeselect.setOrientation(Qt.Horizontal)
        # self.sizeselect.valueChanged.connect(lambda s: self.canvas.set_config('size', s))

        # self.canvas.sliderZ0 = QSlider()
        # self.canvas.sliderZ0.setObjectName('sliderZ0')
        # if USE_SVD:
        #     self.canvas.sliderZ0.setRange(0, 100*MAX_SVD_LAMBDA)
        #     self.canvas.sliderZ0.setSliderPosition(100*MAX_SVD_LAMBDA/2)
        # else:
        #     self.canvas.sliderZ0.setRange(-100,100)
        # self.canvas.sliderZ0.setSingleStep(1)
        # self.canvas.sliderZ0.setOrientation(Qt.Vertical)
        # self.canvas.sliderZ0.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=0,dont_update_undo_list=True))
        # self.canvas.sliderZ0.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.sliderZ0.value() / 100, index=0))
        # self.canvas.sliderZ0.setToolTip('Primary direction gradients magnitude')
        # self.ZToolbar.addWidget(self.canvas.sliderZ0,1,0,4,1)
        # self.canvas.sliderZ1 = QSlider()
        # self.canvas.sliderZ1.setObjectName('sliderZ1')
        # if USE_SVD:
        #     self.canvas.sliderZ1.setRange(0, 100*MAX_SVD_LAMBDA)
        #     self.canvas.sliderZ1.setSliderPosition(100*MAX_SVD_LAMBDA/2)
        # else:
        #     self.canvas.sliderZ1.setRange(-100,100)
        # self.canvas.sliderZ1.setSingleStep(1)
        # self.canvas.sliderZ1.setOrientation(Qt.Vertical)
        # self.canvas.sliderZ1.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=1,dont_update_undo_list=True))
        # self.canvas.sliderZ1.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.sliderZ1.value() / 100, index=1))
        # self.canvas.sliderZ1.setToolTip('Secondary direction gradients magnitude')
        # self.ZToolbar.addWidget(self.canvas.sliderZ1,1,1,4,1)
        # if USE_SVD:
        #     self.canvas.slider_third_channel = QDial()
        #     self.canvas.slider_third_channel.setWrapping(True)
        #     self.canvas.slider_third_channel.setNotchesVisible(True)
        # else:
        #     self.canvas.slider_third_channel = QSlider()
        # self.canvas.slider_third_channel.setObjectName('slider_third_channel')
        # if USE_SVD:
        #     self.canvas.slider_third_channel.setRange(-100*np.pi, 100*np.pi)
        # else:
        #     self.canvas.slider_third_channel.setRange(-100,100)
        # self.canvas.slider_third_channel.setSingleStep(1)
        # self.canvas.slider_third_channel.setOrientation(Qt.Vertical)
        # self.canvas.slider_third_channel.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=2,dont_update_undo_list=True))
        # self.canvas.slider_third_channel.sliderReleased.connect(lambda: self.SetZ_And_Display(value=self.canvas.slider_third_channel.value() / 100, index=2))
        # self.ZToolbar.addWidget(self.canvas.slider_third_channel,1,2,4,4)
        if not self.USE_LAYOUTS_METHOD:
            if self.randomLimitingWeightBox_Enabled:
                self.ZToolbar.addWidget(self.randomLimitingWeightBox,0,10,1,1)
            self.ZToolbar.addWidget(self.DisplayedImageSelection_button,0,0,1,1)
            self.ZToolbar.addWidget(self.CopyFromRandom_button, 0, 1, 1, 1)
            self.ZToolbar.addWidget(self.Copy2Random_button, 0, 2, 1, 1)
            self.ZToolbar.addWidget(self.indicatePeriodicity_button, 0, 3, 1, 1)
            self.ZToolbar.addWidget(self.periodicity_mag_1,0,4,1,1)
            self.ZToolbar.addWidget(self.periodicity_mag_2,0,5,1,1)
        self.ZToolbar2.addWidget(self.special_behavior_button)
        self.ZToolbar2.addWidget(self.IncreaseSTD_button)
        self.ZToolbar2.addWidget(self.DecreaseSTD_button)
        self.ZToolbar2.addWidget(self.STD_increment)
        self.ZToolbar2.addWidget(self.DecreaseTV_button)
        self.ZToolbar2.addWidget(self.ImitateHist_button)
        self.ZToolbar2.addWidget(self.ImitatePatchHist_button)
        self.ZToolbar2.addWidget(self.FoolAdversary_button)
        self.ZToolbar2.addWidget(self.IncreasePeriodicity_1D_button)
        self.ZToolbar2.addWidget(self.IncreasePeriodicity_2D_button)
        self.STD_increment.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        # self.ZToolbar2.addWidget(self.MatchSliders)
        self.ZToolbar2.addWidget(self.ProcessRandZ_button)
        self.ZToolbar2.addWidget(self.ProcessLimitedRandZ_button)

        # Assigning handle to some buttons to canvas:
        self.canvas.special_behavior_button = self.special_behavior_button
        self.canvas.FoolAdversary_button = self.FoolAdversary_button
        self.canvas.selectrect_button = self.selectrect_button
        self.canvas.selectpoly_button = self.selectpoly_button
        self.canvas.actionIncreasePeriodicity_1D = self.IncreasePeriodicity_1D_button
        self.canvas.actionIncreasePeriodicity = self.IncreasePeriodicity_2D_button
        self.canvas.indicatePeriodicity_button = self.indicatePeriodicity_button
        self.canvas.periodicity_mag_1 = self.periodicity_mag_1
        self.canvas.periodicity_mag_1.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.canvas.periodicity_mag_2 = self.periodicity_mag_2
        self.canvas.periodicity_mag_2.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.canvas.apply_scribble_button = self.apply_scribble_button
        self.canvas.loop_apply_scribble_button = self.loop_apply_scribble_button
        self.canvas.Update_Image_Display = self.Update_Image_Display
        self.canvas.Enforce_DT_on_Image_Pair = self.canvas.SR_model.DTE_net.Enforce_DT_on_Image_Pair
        self.canvas.Project_2_Orthog_Nullspace = self.canvas.SR_model.DTE_net.Project_2_ortho_2_NS

        for button_name in IMPRINT_SIZE_CHANGES+IMPRINT_LOCATION_CHANGES:
            cur_button = getattr(self,'%s_imprinting_button'%(button_name))
            # modification_string = cur_button.objectName()[:cur_button.objectName().find('_imprinting_button')]
            # cur_button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification=copy.deepcopy(modification_string)))
            # cur_button[button_name].clicked.connect(lambda:self.Imprinting_modification_Functions(cur_button[-1]))
            setattr(self.canvas,'%s_imprinting_button'%(button_name),cur_button)
        self.up_imprinting_button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification='up'))
        self.down_imprinting_button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification='down'))
        self.right_imprinting_button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification='right'))
        self.left_imprinting_button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification='left'))
        self.wider_imprinting_button.clicked.connect(lambda s: self.canvas.finalize_im_input(modification='wider'))
        self.taller_imprinting_button.clicked.connect(lambda s: self.canvas.finalize_im_input(modification='taller'))
        self.narrower_imprinting_button.clicked.connect(lambda s: self.canvas.finalize_im_input(modification='narrower'))
        self.shorter_imprinting_button.clicked.connect(lambda s: self.canvas.finalize_im_input(modification='shorter'))
        # for button in [getattr(self,'%s_imprinting_button'%(b)) for b in IMPRINT_SIZE_CHANGES+IMPRINT_LOCATION_CHANGES]:
        #     # cur_button = getattr(self,'%s_imprinting_button'%(button_name))
        #     modification_string = button.objectName()[:button.objectName().find('_imprinting_button')]
        #     button.clicked.connect(lambda s:self.canvas.finalize_im_input(modification=copy.deepcopy(modification_string)))
        #     setattr(self.canvas,'%s_imprinting_button'%(button),button)
        # # self.taller_imprinting_button.clicked.connect(lambda s: self.canvas.finalize_im_input(modification='taller'))

        #Scribble:
        self.Scribble_Toolbar.addWidget(self.scribble_reset_button)
        sizeicon = QLabel()
        sizeicon.setPixmap(QPixmap(os.path.join('icons', 'border-weight.png')))
        self.Scribble_Toolbar.addWidget(sizeicon)
        self.sizeselect = QSlider()
        self.sizeselect.setRange(1,20)
        self.sizeselect.setOrientation(Qt.Horizontal)
        self.sizeselect.valueChanged.connect(lambda s: self.canvas.set_config('size', s))
        self.Scribble_Toolbar.addWidget(self.sizeselect)

        self.Scribble_Toolbar.addWidget(self.dropper_button)
        self.Scribble_Toolbar.addWidget(self.pencil_button)
        # self.Scribble_Toolbar.addWidget(self.brush_button)
        self.Scribble_Toolbar.addWidget(self.line_button)
        self.Scribble_Toolbar.addWidget(self.ellipse_button)
        self.Scribble_Toolbar.addWidget(self.polygon_button)
        self.Scribble_Toolbar.addWidget(self.rect_button)
        self.Scribble_Toolbar.addWidget(self.im_input_button)
        self.Scribble_Toolbar.addWidget(self.im_input_auto_location_button)
        self.Scribble_Toolbar.addWidget(self.apply_scribble_button)
        self.Scribble_Toolbar.addWidget(self.loop_apply_scribble_button)
        self.canvas.SelectImage2Display = self.SelectImage2Display
        self.canvas.DisplayedImageSelection_button = self.DisplayedImageSelection_button
        self.canvas.scribble_modes = SCRIBBLE_MODES
        # Scribble mask:
        self.canvas.scribble_mask_canvas = Canvas()
        self.canvas.scribble_mask_canvas.initialize()
        self.canvas.within_drawing = False # I add this to distinguish between first mouse press (initiating polygon drawing) and the rest of the presses. The motivation is to know when I BEGIN a scribble action.
        # Active color display:
        # self.canvas.secondaryButton = QtWidgets.QPushButton(self.Scribble_Toolbar)
        # # self.canvas.secondaryButton.setGeometry(QtCore.QRect(30, 10, 40, 40))
        # self.canvas.secondaryButton.setMinimumSize(QtCore.QSize(40, 40))
        # self.canvas.secondaryButton.setMaximumSize(QtCore.QSize(40, 40))
        # self.canvas.secondaryButton.setText("")
        # self.canvas.secondaryButton.setObjectName("secondaryButton")
        self.canvas.primary_button = QtWidgets.QPushButton(self.Scribble_Toolbar)
        # self.canvas.primaryButton.setGeometry(QtCore.QRect(10, 0, 40, 40))
        self.canvas.primary_button.setMinimumSize(QtCore.QSize(40, 40))
        self.canvas.primary_button.setMaximumSize(QtCore.QSize(40, 40))
        self.canvas.primary_button.setText("")
        self.canvas.primary_button.setObjectName("primary_button")
        self.Scribble_Toolbar.addWidget(self.canvas.primary_button)
        self.canvas.primary_button.pressed.connect(lambda: self.choose_color(self.canvas.set_primary_color))
        self.canvas.set_primary_color('#000000')
        color_state_cycle_icon = QIcon()
        color_state_cycle_icon.addPixmap(QPixmap("icons/color_state_cycle.png"), QIcon.Normal, QIcon.Off)
        self.canvas.cycleColorState_button = QtWidgets.QPushButton(parent=self.Scribble_Toolbar,icon=color_state_cycle_icon)
        self.canvas.cycleColorState_button.setCheckable(False)
        self.canvas.cycleColorState_button.pressed.connect(self.canvas.cycle_color_state)
        self.Scribble_Toolbar.addWidget(self.canvas.cycleColorState_button)
        self.canvas.color_state = 0
        self.canvas.cyclic_color_shift = 0
        self.canvas.local_TV_identifier  = LOCAL_TV_MASK_IDENTIFIERS_RANGE[0]-1
        self.canvas.latest_scribble_color_reset = time.time()

        # self.canvas.primary_color = QColor(Qt.black)

        self.FillShapes_button.clicked.connect(lambda s: self.canvas.set_config('fill', s))
        self.FillShapes_button.setChecked(True)
        # self.open_file(pre_loaded_image=self.image_with_estimated_kernel)
        self.show()

    def Change_kernel_in_use(self,use_estimated=None):
        if use_estimated is None:
            self.using_estimated_kernel = not self.estimatedKenrel_button.isChecked()
        else:
            self.using_estimated_kernel = use_estimated
            self.estimatedKenrel_button.setChecked(use_estimated)
        if self.using_estimated_kernel:
            if self.estimated_kernel is None: #estimate the SR kernel, using the KernelGAN method
                print('Using the KernelGAN method to estimate the SR kernel corresponding to the LR image. This may take a while...')
                KernelGAN_conf = KernelGAN.Config().parse(['--X4'] if self.canvas.DTE_opt['scale']==4 else [])
                KernelGAN_conf.LR_image = self.LR_image
                self.estimated_kernel = KernelGAN.train(KernelGAN_conf)
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
        if checked:
            self.MasksStorage(True)
            self.canvas.desired_im_taken_from_same = not another_image
            if self.canvas.current_display_index == self.canvas.scribble_display_index:
                self.Update_Scribble_Data()
            if another_image:
                path, _ = QFileDialog.getOpenFileName(self, "Desired image for histogram imitation", "",
                                                      "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
                if path:
                    self.canvas.desired_image = data_util.read_img(None, path)
                    # A patch fix - I resize the loaded image to match the HR image dimensions, or else it would change the canvas size irreversibally:
                    self.canvas.desired_image = util.ResizeScribbleImage(self.canvas.desired_image,dsize=tuple(self.canvas.HR_size))
                    if self.canvas.desired_image.ndim>2 and self.canvas.desired_image.shape[2] == 3:
                        self.canvas.desired_image = self.canvas.desired_image[:, :, [2, 1, 0]]
                    else:
                        self.canvas.desired_image = np.repeat(np.expand_dims(self.canvas.desired_image,-1),3,2)
                    im_2_display = 1*self.canvas.desired_image
                    if self.canvas.display_zoom_factor > 1:
                        im_2_display = imresize(im_2_display, self.canvas.display_zoom_factor)
                    pixmap = QPixmap()
                    pixmap.convertFromImage(qimage2ndarray.array2qimage(255*im_2_display))
                    self.canvas.setPixmap(pixmap)
                    # Removed the following line because I now rescale the desired image to canvas size anyway...
                    # self.canvas.setGeometry(QRect(0,0,self.canvas.desired_image.shape[0],self.canvas.desired_image.shape[1]))
                    self.canvas.HR_size = list(self.canvas.desired_image.shape[:2])
            else:
                self.canvas.desired_image = self.canvas.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0)
                # self.canvas.desired_image = qimage2ndarray.rgb_view(self.canvas.pixmap().toImage())/255
            self.canvas.HR_selected_mask = np.ones(self.canvas.desired_image.shape[:2])
            # print('HR mask shape:',self.canvas.HR_selected_mask.shape)
        else:
            # print('HR mask shape:',self.canvas.HR_selected_mask.shape)
            self.canvas.desired_image_HR_mask = 1*self.canvas.HR_selected_mask
            # if another_image:
                # self.canvas.desired_image_HR_mask_vertices = [(self.canvas.display_zoom_factor*cord[0],self.canvas.display_zoom_factor*cord[1]) for cord in self.canvas.HR_mask_vertices]
            self.canvas.desired_image_HR_mask_vertices = self.canvas.HR_mask_vertices
            # else:
            #     self.canvas.desired_image_HR_mask_vertices = self.canvas.HR_mask_vertices_display_size
            self.MasksStorage(False)
            self.Update_Image_Display()
            self.ImitateHist_button.setEnabled(True)
            self.ImitatePatchHist_button.setEnabled(True)
            self.im_input_auto_location_button.setEnabled(True)
            self.im_input_button.setEnabled(True)
            self.canvas.desired_image,self.canvas.desired_image_HR_mask = [self.canvas.desired_image],[self.canvas.desired_image_HR_mask] #Warpping in a list to have a unified framework for the case of transformed hist image versions.
            if DOWNSCALED_HIST_VERSIONS:
                raise Exception('I removed the per-scale temperature modification in the histogram forward function, since I could not tell why it was there.')
                cur_downscaling_factor = DOWNSCALED_HIST_VERSIONS
                while cur_downscaling_factor>MIN_DOWNSCALING_4_HIST:
                    self.canvas.desired_image.append(cv2.resize(self.canvas.desired_image[0],dsize=None,fx=cur_downscaling_factor,fy=cur_downscaling_factor))
                    self.canvas.desired_image_HR_mask.append((cv2.resize(self.canvas.desired_image_HR_mask[0],dsize=None,fx=cur_downscaling_factor,
                        fy=cur_downscaling_factor)>0.5).astype(self.canvas.desired_image_HR_mask[0].dtype))
                    cur_downscaling_factor *= DOWNSCALED_HIST_VERSIONS

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()

        if self.canvas.mode == 'selectrect' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectrect_copy())

        elif self.canvas.mode == 'selectpoly' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectpoly_copy())

        else:
            clipboard.setPixmap(self.canvas.pixmap())

    def Compute_SR_Image(self,dont_update_undo_list=False):
        if self.cur_Z.size(2)==1:
            cur_Z = ((self.cur_Z * torch.ones([1, 1] + self.canvas.Z_size) - 0.5) * 2).type(self.var_L.type())
        else:
            cur_Z = self.cur_Z.type(self.var_L.type())
        self.canvas.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=cur_Z)
        self.canvas.SR_model.netG.eval()
        with torch.no_grad():
            self.canvas.SR_model.fake_H = self.canvas.SR_model.netG(self.canvas.SR_model.model_input)
            if DISPLAY_INDUCED_LR:
                self.induced_LR_image = self.canvas.SR_model.netG.module.DownscaleOP(self.canvas.SR_model.fake_H)
        # if update_default_Z:
        #     self.Update_Default_Z_Image(dont_update_undo_list=dont_update_undo_list)


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
            pixmap_image = qimage2ndarray.array2qimage(self.canvas.Z_mask_display_size*updating_image+(1-self.canvas.Z_mask_display_size)*qimage2ndarray.rgb_view(self.canvas.scribble_mask_canvas.pixmap().toImage()).mean(2))
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
        if chosen_index is not None and chosen_index!=self.canvas.current_display_index:
            self.DisplayedImageSelection_button.setCurrentIndex(chosen_index)# For the case when called not by the DisplayedImageSelectionButton interface
            if self.canvas.current_display_index == self.canvas.scribble_display_index:
                self.Update_Scribble_Data()
            self.canvas.current_display_index = chosen_index
            if self.canvas.current_display_index in [self.canvas.scribble_display_index,self.GT_HR_index,self.ESRGAN_index]:
                self.Zdisplay_button.setChecked(False)
            self.Zdisplay_button.setEnabled(self.canvas.current_display_index not in [self.canvas.scribble_display_index,self.GT_HR_index,self.ESRGAN_index])
            if chosen_index==self.canvas.scribble_display_index:
                self.Reset_Scribbling_Image_Background()
                self.Update_Image_Display()
        # if self.canvas.current_display_index==self.canvas.scribble_display_index:
        #     return
        self.CopyFromRandom_button.setEnabled(self.canvas.current_display_index in self.random_display_indexes)
        if self.canvas.current_display_index in [self.cur_Z_im_index,self.canvas.scribble_display_index]:
            self.canvas.SR_model.fake_H = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
            self.Z_2_display = self.cur_Z
        elif self.canvas.current_display_index in self.random_display_indexes:
            self.canvas.SR_model.fake_H = 1*self.canvas.random_Z_images[self.canvas.current_display_index-self.random_display_indexes[0]+1].unsqueeze(0)
            self.Z_2_display = self.canvas.random_Zs[self.canvas.current_display_index-self.random_display_indexes[0],...].unsqueeze(0)
        else:
            self.Z_2_display = self.no_Z_image
            if self.canvas.current_display_index==self.GT_HR_index:
                self.canvas.SR_model.fake_H = 1*self.GT_HR
            elif self.canvas.current_display_index==self.ESRGAN_index:
                self.canvas.SR_model.fake_H = 1*self.ESRGAN_SR
            elif self.canvas.current_display_index==self.canvas.scribble_display_index:
                pass#I think there is nothing to do here, because I don't assign the image to SR_model.fake_H
        if not self.canvas.current_display_index==self.canvas.scribble_display_index:
            self.Update_Image_Display()
        self.canvas.scribble_undo_redo_enabling(enable=(self.canvas.current_display_index==self.canvas.scribble_display_index))
        self.Z_undo_redo_enabling(enable=(self.canvas.current_display_index==self.cur_Z_im_index))

    def CopyRandom2Default(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        self.cur_Z = (self.canvas.random_Zs[self.canvas.current_display_index-self.random_display_indexes[0],...].to(self.cur_Z.device)*Z_mask+self.cur_Z[0]*(1-Z_mask)).unsqueeze(0)
        self.ReProcess(chosen_display_index=self.cur_Z_im_index)
        self.canvas.Z_optimizer_Reset()
        self.DeriveControlValues()

    def CopyDefault2Random(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        for random_Z_num in range(len(self.canvas.random_Zs)):
            self.canvas.random_Zs[random_Z_num] = self.canvas.random_Zs[random_Z_num]*(1-Z_mask)+self.cur_Z[0]*Z_mask
        self.Process_Z_Alternatives()

    def Process_Z_Alternatives(self):
        stored_Z = 1*self.cur_Z
        for i, random_Z in enumerate(self.canvas.random_Zs):
            self.cur_Z = 1*random_Z.unsqueeze(0)
            self.Compute_SR_Image()
            self.canvas.random_Z_images[i + 1] = self.canvas.SR_model.fake_H[0]
        self.cur_Z = 1*stored_Z
        self.DisplayedImageSelection_button.setEnabled(True)
        self.SelectImage2Display()

    def Process_Random_Z(self,limited):
        if self.num_random_Zs>1 or limited:
            self.Optimize_Z('random_'+('VGG' if VGG_RANDOM_DOMAIN else 'l1')+('_limited' if limited else ''))
        else:
            UNIFORM_RANDOM = False
            Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
            if USE_SVD:
                self.canvas.control_values = Z_mask*torch.stack([self.DrawRandChannel(0,self.max_SVD_Lambda,uniform=UNIFORM_RANDOM),
                    self.DrawRandChannel(0,self.max_SVD_Lambda,uniform=UNIFORM_RANDOM),self.DrawRandChannel(0,np.pi,uniform=UNIFORM_RANDOM)],
                    0).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.control_values
                self.Recompose_cur_Z()
                self.canvas.Update_Z_Sliders()
            else:
                random_Z = (torch.rand([1,self.canvas.SR_model.num_latent_channels]+self.canvas.Z_size)-0.5)*2
                self.cur_Z = Z_mask*random_Z+(1-Z_mask)*self.cur_Z
            self.ReProcess()

    def Validate_Z_optimizer(self,objective):
        # if self.canvas.Z_optimizer is not None:
        if self.latest_optimizer_objective!=objective:# or objective=='hist': # Resetting optimizer in the 'patchhist' case because I use automatic tempersture search there, so I want to search each time for the best temperature.
            self.canvas.Z_optimizer_Reset()

    def MasksStorage(self,store):
        canvas_keys = ['Z_mask','Z_mask_display_size','HR_selected_mask','LR_mask_vertices','HR_size','random_Zs','image_4_scribbling','current_scribble_mask']
        self_keys = ['cur_Z','var_L']
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

    def Crop2BoundingRect(self,arrays,bounding_rect,HR=False):
        operating_on_list = isinstance(arrays,list)
        if not operating_on_list:
            arrays = [arrays]
        if HR:
            bounding_rect = self.canvas.DTE_opt['scale'] * bounding_rect
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
        HR_keys = ['canvas.HR_selected_mask','canvas.SR_model.fake_H','canvas.image_4_scribbling','canvas.current_scribble_mask']+(['canvas.Z_mask','cur_Z'] if self.canvas.HR_Z else [])
        LR_keys = ['var_L']+(['canvas.Z_mask','cur_Z'] if not self.canvas.HR_Z else [])
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

        for key in HR_keys:
            Set_Inner_Attr(key,self.Crop2BoundingRect(Return_Inner_Attr(key),bounding_rect=bounding_rect,HR=True))
            # setattr(self,key, self.Crop2BoundingRect(Return_Inner_Attr(key),bounding_rect=bounding_rect,HR=True))
        for key in LR_keys:
            # setattr(self,key, self.Crop2BoundingRect(Return_Inner_Attr(key),bounding_rect=bounding_rect,HR=False))
            Set_Inner_Attr(key,self.Crop2BoundingRect(Return_Inner_Attr(key),bounding_rect=bounding_rect,HR=False))

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
            self.canvas.SR_model.fake_H = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
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
                self.MasksStorage(True)
                self.Crop_masks_2_BoundingRect(bounding_rect=self.bounding_rect_4_opt)
                self.Z_mask_4_later_merging = torch.from_numpy(self.canvas.Z_mask).type(self.canvas.SR_model.fake_H.dtype).to(self.cur_Z.device)
                data['LR'] = self.var_L
                self.canvas.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=self.Crop2BoundingRect(self.canvas.SR_model.GetLatent(),self.bounding_rect_4_opt,HR=self.canvas.HR_Z))#Because I'm saving initial Z when initializing optimizer
            else:
                self.optimizing_region = False
            self.iters_per_round = 1*ITERS_PER_OPT_ROUND
            if any([phrase in objective for phrase in ['hist','dict','l12GT']]):
                data['HR'] = [torch.from_numpy(np.ascontiguousarray(np.transpose(hist_im, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0) for hist_im in self.canvas.desired_image]
                if 'l1' in objective and self.optimizing_region:
                    data['HR'] = [self.Crop2BoundingRect(hist_im,self.bounding_rect_4_opt,HR=True) for hist_im in data['HR']]
                data['Desired_Im_Mask'] = self.canvas.desired_image_HR_mask
            elif 'desired_SVD' in objective:
                data['desired_Z'] = util.SVD_2_LatentZ(self.canvas.control_values.unsqueeze(0),max_lambda=self.max_SVD_Lambda)
                self.SVD_ValuesStorage(True)
                if self.optimizing_region:
                    data['desired_Z'] = self.Crop2BoundingRect(data['desired_Z'],self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
                    self.canvas.control_values = self.Crop2BoundingRect(self.canvas.control_values.unsqueeze(0),self.bounding_rect_4_opt,HR=self.canvas.HR_Z).squeeze(0)
                self.Set_Extreme_SVD_Values(min_not_max=True)
                data['reference_image_min'] = 1*self.canvas.SR_model.fake_H
                self.Set_Extreme_SVD_Values(min_not_max=False)
                data['reference_image_max'] = 1*self.canvas.SR_model.fake_H
                self.SVD_ValuesStorage(False)
            elif 'periodicity' in objective:
                for p_num in range(len(self.canvas.periodicity_points)):
                    self.canvas.periodicity_points[p_num] = self.canvas.periodicity_points[p_num]*getattr(self,'periodicity_mag_%d'%(p_num+1)).value()/np.linalg.norm(self.canvas.periodicity_points[p_num])
                data['periodicity_points'] = self.canvas.periodicity_points[:2-('1D' in objective)]
            elif 'scribble' in objective:
                data['HR'] = torch.from_numpy(np.ascontiguousarray(np.transpose(self.canvas.image_4_scribbling, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)/255
                data['scribble_mask'] = 1*self.canvas.current_scribble_mask
                data['brightness_factor'] = self.STD_increment.value()#For the brightness increase/decrease functionality
                # self.iters_per_round *= 3
            if any([phrase in objective for phrase in ['STD','Mag']]):
                data['STD_increment'] = self.STD_increment.value()
            initial_Z = 1 * self.cur_Z
            optimization_batch_size = 1
            if self.multiple_inits:
                optimization_batch_size = self.num_random_Zs+(0 if 'random' in objective else 1)
                data['LR'] = data['LR'].repeat([optimization_batch_size,1,1,1]) #When using MULTIPLE_OPT_INITS, optimizing over self.num_random_Zs+1 Zs
                # initial_Z = 1*self.canvas.random_Zs[1:]
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
                    self.canvas.Z_optimizer_logger.append(Logger(self.canvas.DTE_opt,tb_logger_suffix='_%s%s'%(objective,'_%d'%(i) if self.multiple_inits else '')))

            self.canvas.Z_optimizer = Z_optimizer(objective=objective,Z_size=[val*self.canvas.SR_model.Z_size_factor for val in data['LR'].size()[2:]],model=self.canvas.SR_model,
                Z_range=self.max_SVD_Lambda,data=data,initial_LR=self.canvas.Z_optimizer_initial_LR,loggers=self.canvas.Z_optimizer_logger,max_iters=self.iters_per_round,
                image_mask=self.canvas.HR_selected_mask,Z_mask=self.canvas.Z_mask,auto_set_hist_temperature=self.auto_set_hist_temperature,
                batch_size=optimization_batch_size,random_Z_inits=self.random_inits,initial_Z=initial_Z)
            if self.optimizing_region:
                self.MasksStorage(False)
        elif 'random' in objective:
            self.canvas.Z_optimizer.cur_iter = 0
        num_looping_iters = 30 if loop else 1
        for mini_epoch in range(num_looping_iters):
            self.stored_Z = 1 * self.cur_Z
            if self.multiple_inits:
                self.stored_masked_zs = 1*self.canvas.random_Zs
                if 'random' not in objective:
                    self.stored_masked_zs = torch.cat([1*self.cur_Z,self.stored_masked_zs],0)
            else:
                self.stored_masked_zs = 1 * self.cur_Z # Storing previous Z for two reasons: To recover the big picture Z when optimizing_region, and to recover previous Z if loss did not decrease
            try:
                optimization_failed =True
                self.cur_Z = self.canvas.Z_optimizer.optimize()
                optimization_failed = False
            except Exception as e:
                print('Optimization failed: ',e)
                if 'loss' in self.canvas.Z_optimizer.__dict__.keys() and 'bins' in self.canvas.Z_optimizer.loss.__dict__.keys():
                    print('# desired hist images: %d'%(len(self.canvas.desired_image)))
                    print('# Bins: %d, # Image patches: %d'%(self.canvas.Z_optimizer.loss.bins.size(-1),self.canvas.Z_optimizer.loss.patch_extraction_mat.size(1)))
            if optimization_failed or self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1] < 0:
                self.cur_Z = 1 * self.stored_Z
                self.canvas.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=self.cur_Z.type(self.var_L.type()))
                self.SelectImage2Display()
                if loop:
                    break
            else:
                if self.optimizing_region:#            if self.optimizing_region or :
                    temp_Z = (1 * self.cur_Z).to(self.stored_masked_zs.device)
                    self.cur_Z = 1 * self.stored_masked_zs
                    cropping_rect = 1*self.bounding_rect_4_opt
                    if self.canvas.HR_Z:
                        cropping_rect = [self.canvas.DTE_opt['scale']*val for val in self.bounding_rect_4_opt]
                    for Z_num in range(temp_Z.size(0)): # Doing this for the case of finding random Zs far from one another:
                        if ONLY_MODIFY_MASKED_AREA_WHEN_OPTIMIZING:
                            self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]] =\
                                self.Z_mask_4_later_merging*temp_Z[Z_num, ...]+\
                                (1-self.Z_mask_4_later_merging)*self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]]
                        else:
                            self.cur_Z[Z_num, :, cropping_rect[1]:cropping_rect[1] + cropping_rect[3],cropping_rect[0]:cropping_rect[0] + cropping_rect[2]] = temp_Z[Z_num,...]
                if self.multiple_inits:
                    self.canvas.random_Zs = 1*self.cur_Z[-self.num_random_Zs:]
                    self.Process_Z_Alternatives()
                    if 'random' in objective:
                        self.cur_Z = 1*self.stored_Z
                    else:
                        self.cur_Z = 1 * self.cur_Z[0].unsqueeze(0)
                        display_index = ([self.cur_Z_im_index]+self.random_display_indexes)[np.argmin(self.canvas.Z_optimizer.latest_Z_loss_values)]
                        print('Loss values (%d):'%(np.argmin(self.canvas.Z_optimizer.latest_Z_loss_values)),['%.3e'%(val) for val in self.canvas.Z_optimizer.latest_Z_loss_values])
                        self.ReProcess(chosen_display_index=display_index,dont_update_undo_list=mini_epoch<(num_looping_iters-1))
                else:
                    self.DeriveControlValues()
                    self.ReProcess(chosen_display_index=self.cur_Z_im_index if 'scribble' in objective else None,dont_update_undo_list=mini_epoch<(num_looping_iters-1))

            if not optimization_failed:
                print('%d: LR=%.1e, %d iterations: %s loss decreased from %.2e to %.2e by %.2e (factor of %.2e)' % (mini_epoch,self.canvas.Z_optimizer.LR,len(self.canvas.Z_optimizer.loss_values), self.canvas.Z_optimizer.objective,
                    self.canvas.Z_optimizer.loss_values[0],self.canvas.Z_optimizer.loss_values[-1],self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1],
                    self.canvas.Z_optimizer.loss_values[-1]/self.canvas.Z_optimizer.loss_values[0]))
                if (self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))]-self.canvas.Z_optimizer.loss_values[-1])/\
                        np.abs(self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))])<1e-2*self.canvas.Z_optimizer_initial_LR: #If the loss did not decrease, I decrease the optimizer's learning rate
                    self.canvas.Z_optimizer_initial_LR /= 5
                    print('Loss decreased too little relative to beginning, decreasing learning rate to %.3e'%(self.canvas.Z_optimizer_initial_LR))
                    self.canvas.Z_optimizer = None
                    if loop:
                        print('Breaking optimization loop')
                        if mini_epoch<(num_looping_iters-1):
                            self.Add_Z_2_history()
                        break
                else: # This means I'm happy with this optimizer (and its learning rate), so I can cancel the auto-hist-temperature setting, in case it was set to True.
                    self.auto_set_hist_temperature = False
                    self.auto_hist_temperature_mode_button.setChecked(False)

    def DeriveControlValues(self):
        normalized_Z = 1*self.cur_Z.squeeze(0)
        normalized_Z[:2] = (normalized_Z[:2]+self.max_SVD_Lambda)/2/self.max_SVD_Lambda
        normalized_Z[2] /= 2
        new_control_values = torch.stack(util.SVD_Symmetric_2x2(*normalized_Z),0).to( self.canvas.control_values)# Lambda values are not guarnteed to be in [0,self.max_SVD_Lambda], despite Z being limited to [-self.max_SVD_Lambda,self.max_SVD_Lambda].
        self.canvas.derived_controls_indicator = np.logical_or(self.canvas.derived_controls_indicator,self.canvas.Z_mask)
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to( self.canvas.control_values)
        self.canvas.control_values = Z_mask*new_control_values+(1-Z_mask)*self.canvas.control_values
        self.canvas.Update_Z_Sliders()

    def Clear_Z_Mask(self):
        self.canvas.Z_mask = np.ones(self.canvas.Z_size)
        self.canvas.update_Z_mask_display_size()
        self.canvas.HR_selected_mask = np.ones(self.canvas.HR_size)
        if 'current_scribble_mask' in self.canvas.__dict__.keys():
            self.canvas.apply_scribble_button.setEnabled(self.canvas.any_scribbles_within_mask())
            self.canvas.loop_apply_scribble_button.setEnabled(self.canvas.any_scribbles_within_mask())
        self.canvas.LR_mask_vertices = [(0,0),tuple(self.canvas.LR_size[::-1])]
        self.canvas.update_mask_bounding_rect()
        self.canvas.Update_Z_Sliders()
        self.canvas.Z_optimizer_Reset()

    def Invert_Z_Mask(self):
        self.canvas.Z_mask = 1-self.canvas.Z_mask
        self.canvas.update_Z_mask_display_size()
        self.canvas.HR_selected_mask = 1-self.canvas.HR_selected_mask
        self.canvas.Z_optimizer_Reset()
        self.canvas.contained_Z_mask = not self.canvas.contained_Z_mask
        if self.canvas.contained_Z_mask:
            self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.mask_bounding_rect[2:]]))
        else:
            self.FoolAdversary_button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.HR_size]))

    def ApplyUniformZ(self):
        self.canvas.Update_Z_Sliders()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.canvas.control_values.dtype).to(self.canvas.control_values.device)
        self.canvas.control_values = Z_mask * torch.from_numpy(self.canvas.previous_sliders_values).type(Z_mask.dtype).to(Z_mask.device) + (1 - Z_mask) * self.canvas.control_values
        # self.cur_Z = Z_mask * torch.from_numpy(self.canvas.previous_sliders_values).type(Z_mask.dtype) + (1 - Z_mask) * self.cur_Z
        self.Recompose_cur_Z()
        self.ReProcess()
        self.canvas.derived_controls_indicator = (self.canvas.Z_mask*0+(1-self.canvas.Z_mask)*self.canvas.derived_controls_indicator).astype(np.bool)

        # self.canvas.random_Zs[0] = 1 * self.cur_Z.squeeze(0)

    def Recompose_cur_Z(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        new_Z = util.SVD_2_LatentZ(self.canvas.control_values.unsqueeze(0),max_lambda=self.max_SVD_Lambda).to(self.cur_Z.device)
        self.cur_Z = Z_mask * new_Z + (1 - Z_mask) * self.cur_Z
        # self.canvas.random_Zs[0] = 1*self.cur_Z.squeeze(0)

    def SetZ_And_Display(self,value,index,dont_update_undo_list=False):
        self.SetZ(value,index)
        self.Recompose_cur_Z()
        self.ReProcess(dont_update_undo_list=dont_update_undo_list)

    def SetZ(self,value,index,reset_optimizer=True):
    # def SetZ(self,value,index,reset_optimizer=True,recompose_Z=True):
        if reset_optimizer:
            self.canvas.Z_optimizer_Reset()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
        derived_controls_indicator = torch.from_numpy(self.canvas.derived_controls_indicator).type(self.cur_Z.dtype)
        if USE_SVD:
            value_increment = value - 1*self.canvas.previous_sliders_values[index]
            additive_values = torch.from_numpy(value_increment).type(self.canvas.control_values.dtype) + self.canvas.control_values[index].to(derived_controls_indicator.device)
            masked_new_values = Z_mask *(value * (1 - derived_controls_indicator) + derived_controls_indicator * additive_values)
            self.canvas.control_values[index] =  masked_new_values + (1 - Z_mask) * self.canvas.control_values[index].to(derived_controls_indicator.device)
            self.canvas.previous_sliders_values[index] = (1-self.canvas.Z_mask)*self.canvas.previous_sliders_values[index]+self.canvas.ReturnMaskedMapAverage(self.canvas.control_values[index].data.cpu().numpy())
            # if recompose_Z:
            #     self.Recompose_cur_Z()
            if VERBOSITY:
                self.latent_mins = torch.min(torch.cat([self.cur_Z,self.latent_mins],0),dim=0,keepdim=True)[0]
                self.latent_maxs = torch.max(torch.cat([self.cur_Z,self.latent_maxs],0),dim=0,keepdim=True)[0]
                print(self.canvas.lambda0,self.canvas.lambda1,self.canvas.theta)
                print('mins:',[z.item() for z in self.latent_mins.view([-1])])
                print('maxs:', [z.item() for z in self.latent_maxs.view([-1])])
        else:
            raise Exception('Should recode to support Z-mask')
            self.cur_Z[0,index] = value

    def Update_Default_Z_Image(self):
        if 'random_Z_images' in self.canvas.__dict__.keys():
            self.canvas.random_Z_images[0] = 1 * self.canvas.SR_model.fake_H[0]
        else:
            self.canvas.random_Z_images = torch.cat([1*self.canvas.SR_model.fake_H,torch.zeros_like(self.canvas.SR_model.fake_H).repeat([self.num_random_Zs,1,1,1])],0)

    def ToggleDisplay_Z_Image(self,checked):
        for mode in self.canvas.scribble_modes:
            getattr(self,'%s_button'%(mode)).setEnabled(not checked)
        self.Update_Image_Display()
    def Change_Display_Zoom(self,increase):
        # Save scribble data and mask currently on canvas before changing canvas size:
        self.Update_Scribble_Data()
        # Change display zoom factor:
        self.canvas.display_zoom_factor += 1 if increase else -1
        self.DecreaseDisplayZoom_button.setEnabled(self.canvas.display_zoom_factor>DISPLAY_ZOOM_FACTORS_RANGE[0])
        self.IncreaseDisplayZoom_button.setEnabled(self.canvas.display_zoom_factor <DISPLAY_ZOOM_FACTORS_RANGE[1])
        # Resize saved data whose dimensions correspond to display size:
        self.canvas.image_4_scribbling_display_size = \
            util.ResizeScribbleImage(self.canvas.image_4_scribbling_display_size,dsize=tuple([self.canvas.display_zoom_factor * val for val in self.canvas.HR_size]))
        self.Update_Scribble_Mask_Canvas(initialize=True)
        self.canvas.update_Z_mask_display_size()
        # Apply new size image to canvas:
        self.Update_Image_Display()
        # Update canvas window size and title:
        self.Update_Canvas_Size_and_Title()

    def Update_Canvas_Size_and_Title(self):
        top_left_corner = self.canvas.geometry().getCoords()[:2]
        self.canvas.setGeometry(QRect(top_left_corner[0], top_left_corner[1], self.canvas.display_zoom_factor * self.canvas.HR_size[1],
                  self.canvas.display_zoom_factor * self.canvas.HR_size[0]))
        self.canvas.setWindowTitle(self.image_name+(' (x%d)'%(self.canvas.display_zoom_factor) if self.canvas.display_zoom_factor>1 else ''))


    def Update_Image_Display(self):
        pixmap = QPixmap()
        if self.Zdisplay_button.isChecked():
            im_2_display = 255/2/self.max_SVD_Lambda*(self.max_SVD_Lambda+self.Z_2_display[0].data.cpu().numpy().transpose(1,2,0)).copy()
        else:
            if self.canvas.current_display_index==self.canvas.scribble_display_index:
                im_2_display = 1*self.canvas.image_4_scribbling_display_size
            else:
                im_2_display = 255 * self.canvas.SR_model.fake_H.detach()[0].float().cpu().numpy().transpose(1, 2, 0).copy()
        if self.canvas.display_zoom_factor>1 and not ((not self.Zdisplay_button.isChecked()) and self.canvas.current_display_index==self.canvas.scribble_display_index):
            # For the specific case of updating scribble image, image is allready in correct size
            im_2_display = imresize(im_2_display,self.canvas.display_zoom_factor)
        pixmap.convertFromImage(qimage2ndarray.array2qimage(im_2_display))
        self.canvas.setPixmap(pixmap)
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
        path, _ = QFileDialog.getOpenFileName(self, "Open Z file to deduce editing mask", "",
                                              "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
        if path:
            loaded_Z = data_util.read_img(None, path)
            edited_pixels_map = np.any(loaded_Z!=127/255,axis=2)
            self.canvas.HR_selected_mask = 1*edited_pixels_map
            self.canvas.FoolAdversary_button.setEnabled(False)
            self.canvas.contained_Z_mask = False
            assert self.canvas.HR_Z,'Not supprting other option'
            self.canvas.Z_mask = 1*edited_pixels_map
            self.canvas.update_Z_mask_display_size()
            self.canvas.Update_Z_Sliders()
            self.canvas.Z_optimizer_Reset()
            self.canvas.apply_scribble_button.setEnabled(self.canvas.any_scribbles_within_mask())
            self.canvas.loop_apply_scribble_button.setEnabled(self.canvas.any_scribbles_within_mask())

    def Initialize_SR_model(self,kernel=None,reprocess=True):
        self.canvas.SR_model = create_model(self.opt, init_Dnet=False, init_Fnet=VGG_RANDOM_DOMAIN,kernel=kernel)
        self.canvas.Z_optimizer_Reset()
        if reprocess:
            self.ReProcess()

    def Load_Z(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Z file", "","PNG image files (*_Z.png)")
        if path:
            loaded_Z = data_util.read_img(None, path)
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

            self.canvas.Z_mask = 1*stored_mask
            self.canvas.update_Z_mask_display_size()
            self.canvas.Z_optimizer_Reset()

    def Estimate_DerivedControlIndicator(self):
        PATCH_SIZE_4_ESTIMATION = 3
        patch_extraction_map = ReturnPatchExtractionMat(self.canvas.Z_mask,PATCH_SIZE_4_ESTIMATION,device=self.cur_Z.device,patches_overlap=1)
        STD_map = torch.sparse.mm(patch_extraction_map, self.cur_Z.mean(dim=1).view([-1, 1])).view([PATCH_SIZE_4_ESTIMATION ** 2, -1]).std(dim=0).view(
            [val-PATCH_SIZE_4_ESTIMATION+1 for val in list(self.cur_Z.size()[2:])])
        return np.pad((STD_map>0).data.cpu().numpy().astype(np.bool),pad_width=int(PATCH_SIZE_4_ESTIMATION//2),mode='edge')

    def open_file(self,pre_loaded_image=None,HR_image=False):
        """
        Open image file for editing, scaling the smaller dimension and cropping the remainder.
        :return:
        """
        loaded_image = None
        if not pre_loaded_image:
            path, _ = QFileDialog.getOpenFileName(self,"Open GT HR file" if HR_image else "Open file", "", "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
            if path:
                self.image_name = path.split('/')[-1].split('.')[0]
                loaded_image = data_util.read_img(None, path)
                if loaded_image.ndim>2 and loaded_image.shape[2] == 3:
                    loaded_image = loaded_image[:, :, [2, 1, 0]]
        else:
            loaded_image = pre_loaded_image

        if loaded_image is not None:
            if HR_image:
                SR_scale = self.canvas.DTE_opt['scale']
                loaded_image = loaded_image[:loaded_image.shape[0]//SR_scale*SR_scale,:loaded_image.shape[1]//SR_scale*SR_scale,:] #Removing bottom right margins to make the image shape adequate to this SR factor
                # if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
                self.GT_HR = torch.from_numpy(np.transpose(loaded_image, (2, 0, 1))).float().to(self.canvas.SR_model.device).unsqueeze(0)
                # else:
                #     pixmap = QPixmap()
                #     pixmap.convertFromImage(qimage2ndarray.array2qimage(255 * loaded_image))
                #     self.GT_canvas.setPixmap(pixmap)
                self.canvas.HR_size = list(self.GT_HR.size()[2:])
                self.var_L = self.canvas.SR_model.netG.module.DownscaleOP(torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0))
                self.LR_image = np.clip(255*self.var_L[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8)
                self.DisplayedImageSelection_button.model().item(self.GT_HR_index).setEnabled(True)
            else:
                self.var_L = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.canvas.SR_model.device).unsqueeze(0)
                self.LR_image = (255*loaded_image).astype(np.uint8)
                self.DisplayedImageSelection_button.model().item(self.GT_HR_index).setEnabled(False)
            if self.display_ESRGAN:
                ESRGAN_opt = option.parse('./options/test/GUI_esrgan.json', is_train=False,name='RRDB_ESRGAN_x4')
                ESRGAN_opt['name']
                ESRGAN_opt = option.dict_to_nonedict(ESRGAN_opt)
                ESRGAN_opt['network_G']['latent_input'] = 'None'
                ESRGAN_opt['network_G']['latent_channels'] = 0
                ESRGAN_opt['network_G']['DTE_arch'] = 0
                ESRGAN_model = create_model(ESRGAN_opt)
                ESRGAN_model.netG.eval()
                with torch.no_grad():
                    self.ESRGAN_SR = ESRGAN_model.netG(self.var_L).detach().to(torch.device('cpu'))
                self.canvas.HR_size = list(self.ESRGAN_SR.size()[2:])

            if 'random_Z_images' in self.canvas.__dict__.keys():
                del self.canvas.random_Z_images
            self.canvas.LR_size = list(self.var_L.size()[2:])
            self.canvas.Z_size = [val*self.canvas.DTE_opt['scale'] for val in self.canvas.LR_size] if self.canvas.HR_Z else self.canvas.LR_size
            self.canvas.Z_mask = np.ones(self.canvas.Z_size)
            self.canvas.update_Z_mask_display_size()
            self.canvas.derived_controls_indicator = np.zeros(self.canvas.Z_size)
            self.cur_Z = torch.zeros(size=[1,self.canvas.SR_model.num_latent_channels]+self.canvas.Z_size).to(self.canvas.SR_model.device)
            self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1]).to(self.cur_Z.device)
            self.canvas.previous_sliders_values = \
                np.array([self.canvas.sliderZ0.value(),self.canvas.sliderZ1.value(),self.canvas.slider_third_channel.value()]).reshape([3,1,1])/100*np.ones([1]+self.canvas.Z_size)
            if USE_SVD:
                self.canvas.control_values = 0.5*torch.ones_like(self.cur_Z).squeeze(0)
                # self.SetZ(0.5*MAX_SVD_LAMBDA, 0,recompose_Z=False)
                # self.SetZ(0.5*MAX_SVD_LAMBDA, 1,recompose_Z=False)
                # self.SetZ(0.5, 2)
                self.SetZ(0.5*self.max_SVD_Lambda, 0)
                self.SetZ(0.5*self.max_SVD_Lambda, 1)
                self.SetZ(0.5, 2)
                self.Recompose_cur_Z()
                if VERBOSITY:
                    self.latent_mins = 100 * torch.ones([1, 3, 1, 1])
                    self.latent_maxs = -100 * torch.ones([1, 3, 1, 1])
            # Reset some stuff:
            if self.using_estimated_kernel: # Need to reset model to default kernel:
                self.using_estimated_kernel = False
                # self.canvas.SR_model = create_model(self.opt, init_Dnet=False, init_Fnet=VGG_RANDOM_DOMAIN,kernel='reset_2_default')
                self.estimatedKenrel_button.setChecked(False)
                # self.Initialize_SR_model(kernel='reset_2_default',reprocess=False) #Not recomputing just for efficiency, as ReProcess is called soon.
            self.estimatedKenrel_button.setEnabled((not HR_image) and (self.canvas.DTE_opt['scale'] in [2,4])) #KernelGAN only supprot 2x and 4x SR. For synthetically downscaled HR images, there is no need to estimated the kernel.
            self.estimated_kernel = None
            self.Z_history = deque(maxlen=Z_HISTORY_LENGTH)
            self.Z_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
            self.canvas.scribble_history = deque(maxlen=Z_HISTORY_LENGTH)
            self.canvas.scribble_mask_history = deque(maxlen=Z_HISTORY_LENGTH)
            self.canvas.scribble_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
            self.canvas.scribble_mask_redo_list = deque(maxlen=Z_HISTORY_LENGTH)
            self.Initialize_SR_model(kernel='reset_2_default')
            # self.ReProcess()
            self.saved_outputs_counter = 0
            # self.canvas.random_Z_images = torch.cat([self.canvas.SR_model.fake_H,torch.zeros_like(self.canvas.SR_model.fake_H).repeat([self.num_random_Zs,1,1,1])],0)
            self.DisplayedImageSelection_button.setCurrentIndex(self.cur_Z_im_index)
            # if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
            self.no_Z_image = torch.from_numpy(np.transpose(2 * (resize(image=data_util.read_img(None, 'icons/X.png')[:, :, ::-1], output_shape=self.canvas.HR_size) - 0.5), (2, 0, 1))).float().to(self.canvas.SR_model.device).unsqueeze(0)
            # else:
            #     self.canvas.HR_size = list(self.canvas.SR_model.fake_H.size()[2:])
            if 'current_scribble_mask' in self.canvas.__dict__.keys():
                del self.canvas.current_scribble_mask
            self.Clear_Z_Mask()
            self.canvas.image_4_scribbling = None
            self.Reset_Image_4_Scribbling()
            self.canvas.show()
            self.Update_Canvas_Size_and_Title()

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

    def save_file(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "PNG Image file (*.png)")

        if path:
            imageio.imsave(path,np.clip(255*self.canvas.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0),0,255).astype(np.uint8))
            # pixmap = self.canvas.pixmap()
            # pixmap.save(path, "PNG" )

    def save_file_and_Z_map(self):
        """
        Save active canvas and cur_Z map to image file.
        :return:
        """
        while True:
            path = os.path.join('/'.join(self.canvas.DTE_opt['path']['results_root'].split('/')[:-2]),'GUI_outputs','%s_%d%s.png'%(self.image_name,self.saved_outputs_counter,'%s'))
            if not os.path.isfile(path%('')):
                break
            self.saved_outputs_counter += 1
        # path, _ = QFileDialog.getSaveFileName(self, "Save current image, Z and scribble data", path,
        #                                       "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
        if path:
            imageio.imsave(path%(''),np.clip(255*self.canvas.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0),0,255).astype(np.uint8))
            imageio.imsave(path%('_Z'),np.clip(255/2/self.max_SVD_Lambda*(self.max_SVD_Lambda+self.cur_Z[0].data.cpu().numpy().transpose(1,2,0)),0,255).astype(np.uint8))
            # if DISPLAY_INDUCED_LR:
            if LR_INTERPOLATION_4_SAVING=='NN':
                # interpolated_LR = cv2.resize(np.clip(255*self.induced_LR_image[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8),
                #     dsize=tuple(self.canvas.HR_size[::-1]),interpolation=cv2.INTER_NEAREST)
                interpolated_LR = cv2.resize(self.LR_image,dsize=tuple(self.canvas.HR_size[::-1]), interpolation=cv2.INTER_NEAREST)
            imageio.imsave(path.replace('_%d'%(self.saved_outputs_counter),'') % ('_LR'), interpolated_LR)
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
            print('Saved image %s'%(path%('')))
            self.saved_outputs_counter += 1

    def invert(self):
        img = QImage(self.canvas.pixmap())
        img.invertPixels()
        pixmap = QPixmap()
        pixmap.convertFromImage(img)
        self.canvas.setPixmap(pixmap)

    def flip_horizontal(self):
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(-1, 1)))

    def flip_vertical(self):
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(1, -1)))



if __name__ == '__main__':

    app = QApplication([])
    window = MainWindow()
    app.exec_()