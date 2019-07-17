from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from MainWindow import Ui_MainWindow

import os
import random
import types

# EDitable SR imports:
from models import create_model
import options.options as option
import utils.util as util
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
import copy

BRUSH_MULT = 3
SPRAY_PAINT_MULT = 5
SPRAY_PAINT_N = 100
USE_SVD = True
VERBOSITY = False
MAX_SVD_LAMBDA = 1.
Z_OPTIMIZER_INITIAL_LR = 1e-1
Z_OPTIMIZER_INITIAL_LR_4_RANDOM = 1e-1
NUM_RANDOM_ZS = 3
VGG_RANDOM_DOMAIN = False
DISPLAY_GT_HR = True
DISPLAY_ESRGAN_RESULTS = True
ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS = True
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
MULTIPLE_OPT_INITS = False
AUTO_CYCLE_LENGTH_4_PERIODICITY = True

assert not (DICTIONARY_REPLACES_HISTOGRAM and L1_REPLACES_HISTOGRAM)

COLORS = [
    '#000000', '#82817f', '#820300', '#868417', '#007e03', '#037e7b', '#040079',
    '#81067a', '#7f7e45', '#05403c', '#0a7cf6', '#093c7e', '#7e07f9', '#7c4002',

    '#ffffff', '#c1c1c1', '#f70406', '#fffd00', '#08fb01', '#0bf8ee', '#0000fa',
    '#b92fc2', '#fffc91', '#00fd83', '#87f9f9', '#8481c4', '#dc137d', '#fb803c',
]

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]

MODES = [
    'selectpoly', 'selectrect','indicatePeriodicity',
    #'eraser', 'fill',
    #'dropper', 'stamp',
    'pen',
    #'spray', 'text',
    #'line', #'rand_Z',#'polyline',
    #'rect',
    #'polygon',
    #'ellipse', 'roundrect'
]

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

    primary_color = QColor(Qt.black)
    secondary_color = None

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

    # Mouse events.

    def mousePressEvent(self, e):
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)

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
        self.LR_mask_vertices = [(int(np.round(p.x()/self.DTE_opt['scale'])),int(np.round(p.y()/self.DTE_opt['scale']))) for p in (self.history_pos + [self.current_pos])]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        HR_mask_vertices = [(coord[0]*self.DTE_opt['scale'],coord[1]*self.DTE_opt['scale']) for coord in self.LR_mask_vertices]
        self.HR_selected_mask = cv2.fillPoly(self.HR_selected_mask,[np.array(HR_mask_vertices)],(1,1,1))
        self.Z_mask = np.zeros(self.Z_size)
        if self.HR_Z:
            self.Z_mask = cv2.fillPoly(self.Z_mask, [np.array(HR_mask_vertices)], (1, 1, 1))
        else:
            self.Z_mask = cv2.fillPoly(self.Z_mask,[np.array(self.LR_mask_vertices)],(1,1,1))
        self.Update_Z_Sliders()
        self.Z_optimizer_Reset()
        self.selectpolyButton.setChecked(False)
        # self.selectpoly_copy()#I add this to remove the dashed selection lines from the image, after I didn't find any better way. This removes it if done immediatly after selection, for some yet to be known reason

    def indicatePeriodicity_mousePressEvent(self, e):
        if not self.locked or e.button == Qt.RightButton:
            # self.periodicity_painter = None
            # self.periodicity_points_pen = None
            self.active_shape_fn = 'drawPolygon'
            self.preview_pen = SELECTION_PEN
            self.generic_poly_mousePressEvent(e)
        if len(self.history_pos)==3:
            self.locked = True
            # self.generic_poly_mousePressEvent(e)
            self.actionIncreasePeriodicity.setEnabled(True)
            self.actionIncreasePeriodicity_1D.setEnabled(True)
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
            # self.indicatePeriodicityButton.setDown(False)
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

    def selectrect_mousePressEvent(self, e):
        self.active_shape_fn = 'drawRect'
        self.preview_pen = SELECTION_PEN
        self.generic_shape_mousePressEvent(e)

    def selectrect_timerEvent(self, final=False):
        self.generic_shape_timerEvent(final)

    def selectrect_mouseMoveEvent(self, e):
        if not self.locked:
            self.current_pos = e.pos()
    def update_mask_bounding_rect(self):
        self.mask_bounding_rect = np.array(cv2.boundingRect(np.stack([list(p) for p in self.LR_mask_vertices], 1).transpose()))
        self.FoolAdversary_Button.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.mask_bounding_rect[2:]]))
        self.contained_Z_mask = True

    def selectrect_mouseReleaseEvent(self, e):
        self.current_pos = e.pos()
        self.locked = True
        self.HR_selected_mask = np.zeros(self.HR_size)
        self.LR_mask_vertices = [(int(np.round(p.x()/self.DTE_opt['scale'])),int(np.round(p.y()/self.DTE_opt['scale']))) for p in [self.origin_pos, self.current_pos]]
        if not self.in_picking_desired_hist_mode:
            self.update_mask_bounding_rect()
        HR_mask_vertices = [(coord[0]*self.DTE_opt['scale'],coord[1]*self.DTE_opt['scale']) for coord in self.LR_mask_vertices]
        self.HR_selected_mask = cv2.rectangle(self.HR_selected_mask,HR_mask_vertices[0],HR_mask_vertices[1],(1,1,1),cv2.FILLED)
        self.Z_mask = np.zeros(self.Z_size)
        if self.HR_Z:
            self.Z_mask = cv2.rectangle(self.Z_mask, HR_mask_vertices[0], HR_mask_vertices[1], (1, 1, 1),cv2.FILLED)
        else:
            self.Z_mask = cv2.rectangle(self.Z_mask,self.LR_mask_vertices[0],self.LR_mask_vertices[1],(1,1,1),cv2.FILLED)
        self.Update_Z_Sliders()
        self.Z_optimizer_Reset()
        self.selectrectButton.setChecked(False)#This does not work, probably because of some genral property set for all "mode" buttons.
        # self.selectrect_copy()  # I add this to remove the dashed selection lines from the image, after I didn't find any better way. This removes it if done immediatly after selection, for some yet to be known reason

    def Z_optimizer_Reset(self):
        self.Z_optimizer_initial_LR = Z_OPTIMIZER_INITIAL_LR
        self.Z_optimizer = None
        self.Z_optimizer_logger = None

    def ReturnMaskedMapAverage(self,map):
        return np.sum(map*self.Z_mask)/np.sum(self.Z_mask)

    def Update_Z_Sliders(self):
        # self.sliderZ0.setSliderPosition(100*np.sum(self.lambda0.data.cpu().numpy()*self.Z_mask)/np.sum(self.Z_mask))
        # self.sliderZ1.setSliderPosition(100*np.sum(self.lambda1.data.cpu().numpy()*self.Z_mask)/np.sum(self.Z_mask))
        # self.slider_third_channel.setSliderPosition(100*np.sum(self.theta.data.cpu().numpy()*self.Z_mask)/np.sum(self.Z_mask))
        sliderZ0_new_val = self.ReturnMaskedMapAverage(self.control_values[0].data.cpu().numpy())
        sliderZ1_new_val = self.ReturnMaskedMapAverage(self.control_values[1].data.cpu().numpy())
        slider_third_channel_new_val = self.ReturnMaskedMapAverage(self.control_values[2].data.cpu().numpy())
        self.sliderZ0.setSliderPosition(100*sliderZ0_new_val)
        self.sliderZ1.setSliderPosition(100*sliderZ1_new_val)
        self.slider_third_channel.setSliderPosition(100*slider_third_channel_new_val)
        # self.previous_sliders_values = np.expand_dims(self.Z_mask,0)*np.ones_like(self.control_values.data.cpu().numpy()) +\
        #     np.expand_dims(1-self.Z_mask, 0)*self.previous_sliders_values
        # self.previous_sliders_values = np.expand_dims(self.Z_mask,0)*self.control_values.data.cpu().numpy() +np.expand_dims(1-self.Z_mask, 0)*self.previous_sliders_values
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

    def pen_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, self.config['size'], Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            scribble_mask = QPainter(self.scribble_mask.pixmap())
            scribble_mask.setPen(QPen(QColor(1,1,1), self.config['size'], Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            scribble_mask.drawLine(self.last_pos, e.pos())
            self.scribble_mask.update()
            self.last_pos = e.pos()
            self.update()

    def pen_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Brush events

    def brush_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def brush_mouseMoveEvent(self, e):
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, self.config['size'] * BRUSH_MULT, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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
            p.setPen(QPen(self.active_color, 1))

            for n in range(self.config['size'] * SPRAY_PAINT_N):
                xo = random.gauss(0, self.config['size'] * SPRAY_PAINT_MULT)
                yo = random.gauss(0, self.config['size'] * SPRAY_PAINT_MULT)
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

            p = QPainter(self.pixmap())
            p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))

            if self.config['fill']:
                p.setBrush(QBrush(self.secondary_color))
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
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
            p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            p.drawLine(self.origin_pos, e.pos())
            self.update()

        self.reset_mode()

    # Generic poly events
    def generic_poly_mousePressEvent(self, e):
        SHOW_CHOSEN_POINTS = False
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
        self.timer_cleanup()
        p = QPainter(self.pixmap())
        p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        # Note the brush is ignored for polylines.
        if self.secondary_color:
            p.setBrush(QBrush(self.secondary_color))

        getattr(p, self.active_shape_fn)(*self.history_pos + [e.pos()])
        self.update()
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
        self.setupUi(self)

        # Editable SR:
        opt = option.parse('./options/test/GUI_esrgan.json', is_train=False)
        opt = option.dict_to_nonedict(opt)
        self.SR_model = create_model(opt,init_Dnet=True,init_Fnet=VGG_RANDOM_DOMAIN)
        matplotlib.use('Qt5Agg')
        matplotlib.interactive(True)

        self.saved_outputs_counter = 0
        self.desired_hist_image = None
        self.auto_set_hist_temperature = False

        # Replace canvas placeholder from QtDesigner.
        self.horizontalLayout.removeWidget(self.canvas)
        self.canvas = Canvas()
        self.canvas.Z_optimizer_Reset()
        self.latest_optimizer_objective = ''
        self.canvas.DTE_opt = opt
        self.canvas.initialize()
        self.canvas.HR_Z = 'HR' in self.canvas.DTE_opt['network_G']['latent_input_domain']

        # Scribble mask:
        self.canvas.scribble_mask = Canvas()
        self.canvas.scribble_mask.initialize()

        # We need to enable mouse tracking to follow the mouse without the button pressed.
        self.canvas.setMouseTracking(True)
        # Enable focus to capture key inputs.
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.horizontalLayout.addWidget(self.canvas)
        if not ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
            if DISPLAY_GT_HR:
                #Add a 2nd canvas:
                self.GT_canvas = Canvas()
                self.GT_canvas.initialize()
                self.horizontalLayout.addWidget(self.GT_canvas)
            if DISPLAY_ESRGAN_RESULTS:
                self.ESRGAN_canvas = Canvas()
                self.ESRGAN_canvas.initialize()
                self.horizontalLayout.addWidget(self.ESRGAN_canvas)

        if DISPLAY_INDUCED_LR:
            #Add a 3rd canvas:
            self.LR_canvas = Canvas()
            self.LR_canvas.initialize()
            self.horizontalLayout.addWidget(self.LR_canvas)

        # Setup the mode buttons
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        for mode in MODES:
            btn = getattr(self, '%sButton' % mode)
            btn.pressed.connect(lambda mode=mode: self.canvas.set_mode(mode))
            mode_group.addButton(btn)

        # Setup up action signals
        self.actionCopy.triggered.connect(self.copy_to_clipboard)

        # Initialize animation timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.on_timer)
        self.timer.setInterval(100)
        self.timer.start()

        # Menu options
        self.actionNewImage.triggered.connect(self.canvas.initialize)
        self.actionOpenImage.triggered.connect(self.open_file)
        self.actionLoad_Z.triggered.connect(self.Load_Z)


        self.actionProcessRandZ.triggered.connect(lambda x: self.Process_Random_Z(limited=False))
        self.actionProcessLimitedRandZ.triggered.connect(lambda x: self.Process_Random_Z(limited=True))
        # self.DisplayedImageSelectionButton.currentIndexChanged.connect(self.PickRandom_Z)
        self.DisplayedImageSelectionButton.highlighted.connect(self.SelectImage2Display)
        if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
            if DISPLAY_ESRGAN_RESULTS:
                self.DisplayedImageSelectionButton.addItem('ESRGAN')
                self.DisplayedImageSelectionButton.setEnabled(True)
                self.ESRGAN_index = self.DisplayedImageSelectionButton.findText('ESRGAN')
            if DISPLAY_GT_HR:
                self.DisplayedImageSelectionButton.addItem('GT')
                self.DisplayedImageSelectionButton.setEnabled(True)
                self.GT_HR_index = self.DisplayedImageSelectionButton.findText('GT')
        self.DisplayedImageSelectionButton.addItem('Z')
        self.cur_Z_im_index = self.DisplayedImageSelectionButton.findText('Z')
        self.canvas.current_random_Z_index = 1*self.cur_Z_im_index
        self.DisplayedImageSelectionButton.addItems([str(i+1) for i in range(self.num_random_Zs)])
        self.actionCopyFromRandom.triggered.connect(self.CopyRandom2Default)
        self.actionCopy2Random.triggered.connect(self.CopyDefault2Random)
        self.actionIncreaseSTD.triggered.connect(lambda x:self.Optimize_Z('STD_increase' if RELATIVE_STD_OPT else 'max_STD'))
        self.actionDecreaseSTD.triggered.connect(lambda x:self.Optimize_Z('STD_decrease' if RELATIVE_STD_OPT else 'min_STD'))
        self.actionDecreaseTV.triggered.connect(lambda x:self.Optimize_Z('TV'))
        self.actionImitateHist.triggered.connect(lambda x:self.Optimize_Z('hist'))
        self.actionImitatePatchHist.triggered.connect(lambda x:self.Optimize_Z('patchhist'))
        self.actionFoolAdversary.triggered.connect(lambda x:self.Optimize_Z('Adversarial'))
        self.actionIncreasePeriodicity.triggered.connect(lambda x:self.Optimize_Z('periodicity'))
        self.actionIncreasePeriodicity_1D.triggered.connect(lambda x:self.Optimize_Z('periodicity_1D'))
        self.actionMatchSliders.triggered.connect(lambda x:self.Optimize_Z('desired_SVD'))

        self.UnselectButton.clicked.connect(self.Clear_Z_Mask)
        self.invertSelectionButton.clicked.connect(self.Invert_Z_Mask)
        self.uniformZButton.clicked.connect(self.ApplyUniformZ)
        self.desiredHistModeButton.clicked.connect(lambda checked: self.DesiredHistMode(checked,another_image=False))
        self.ZdisplayButton.clicked.connect(self.ToggleDisplay_Z_Image)
        self.desiredImageHistModeButton.clicked.connect(lambda checked: self.DesiredHistMode(checked,another_image=True))
        self.canvas.in_picking_desired_hist_mode = False
        self.auto_hist_temperature_mode_button.clicked.connect(lambda checked:self.AutoHistTemperatureMode(checked))

        # self.actionReProcess.triggered.connect(self.ReProcess)

        self.actionSaveImage.triggered.connect(self.save_file)
        self.actionAutoSaveImage.triggered.connect(self.save_file_and_Z_map)
        self.actionClearImage.triggered.connect(self.canvas.reset)
        self.actionInvertColors.triggered.connect(self.invert)
        self.actionFlipHorizontal.triggered.connect(self.flip_horizontal)
        self.actionFlipVertical.triggered.connect(self.flip_vertical)

        sizeicon = QLabel()
        sizeicon.setPixmap(QPixmap(os.path.join('images', 'border-weight.png')))
        # self.drawingToolbar.addWidget(sizeicon)
        self.sizeselect = QSlider()
        self.sizeselect.setRange(1,20)
        self.sizeselect.setOrientation(Qt.Horizontal)
        self.sizeselect.valueChanged.connect(lambda s: self.canvas.set_config('size', s))
        self.canvas.sliderZ0 = QSlider()
        self.canvas.sliderZ0.setObjectName('sliderZ0')
        if USE_SVD:
            self.canvas.sliderZ0.setRange(0, 100*MAX_SVD_LAMBDA)
            self.canvas.sliderZ0.setSliderPosition(100*MAX_SVD_LAMBDA/2)
        else:
            self.canvas.sliderZ0.setRange(-100,100)
        self.canvas.sliderZ0.setSingleStep(1)
        self.canvas.sliderZ0.setOrientation(Qt.Vertical)
        self.canvas.sliderZ0.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=0))
        self.canvas.sliderZ0.sliderReleased.connect(self.Remember_Zmap)
        self.ZToolbar.addWidget(self.canvas.sliderZ0)
        self.canvas.sliderZ1 = QSlider()
        self.canvas.sliderZ1.setObjectName('sliderZ1')
        if USE_SVD:
            self.canvas.sliderZ1.setRange(0, 100*MAX_SVD_LAMBDA)
            self.canvas.sliderZ1.setSliderPosition(100*MAX_SVD_LAMBDA/2)
        else:
            self.canvas.sliderZ1.setRange(-100,100)
        self.canvas.sliderZ1.setSingleStep(1)
        self.canvas.sliderZ1.setOrientation(Qt.Vertical)
        self.canvas.sliderZ1.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=1))
        self.canvas.sliderZ1.sliderReleased.connect(self.Remember_Zmap)
        self.ZToolbar.addWidget(self.canvas.sliderZ1)
        if USE_SVD:
            self.canvas.slider_third_channel = QDial()
            self.canvas.slider_third_channel.setWrapping(True)
            self.canvas.slider_third_channel.setNotchesVisible(True)
        else:
            self.canvas.slider_third_channel = QSlider()
        self.canvas.slider_third_channel.setObjectName('slider_third_channel')
        if USE_SVD:
            self.canvas.slider_third_channel.setRange(-100*np.pi, 100*np.pi)
        else:
            self.canvas.slider_third_channel.setRange(-100,100)
        self.canvas.slider_third_channel.setSingleStep(1)
        self.canvas.slider_third_channel.setOrientation(Qt.Vertical)
        self.canvas.slider_third_channel.sliderMoved.connect(lambda s: self.SetZ_And_Display(value=s / 100, index=2))
        # self.canvas.slider_third_channel.sliderReleased.connect(lambda s=self.canvas.slider_third_channel.sliderPosition():self.SetZ(value=self.canvas.slider_third_channel.sliderPosition()/100,index=2))
        self.canvas.slider_third_channel.sliderReleased.connect(self.Remember_Zmap)
        self.ZToolbar.addWidget(self.canvas.slider_third_channel)
        # self.ZToolbar.addAction(self.actionProcessRandZ)
        # self.ZToolbar.addAction(self.actionProcessLimitedRandZ)
        if self.randomLimitingWeightBox_Enabled:
            self.ZToolbar.addWidget(self.randomLimitingWeightBox)
        self.ZToolbar.addWidget(self.DisplayedImageSelectionButton)
        self.ZToolbar.addAction(self.actionCopyFromRandom)
        # self.ZToolbar.setIconSize(QSize(30,30))
        self.ZToolbar.addAction(self.actionCopy2Random)
        self.ZToolbar.addWidget(self.indicatePeriodicityButton)
        self.ZToolbar.insertSeparator(self.actionProcessRandZ)
        self.ZToolbar.addWidget(self.periodicity_mag_1)
        self.ZToolbar.addWidget(self.periodicity_mag_2)
        self.ZToolbar2.addAction(self.actionIncreaseSTD)
        self.ZToolbar2.addAction(self.actionDecreaseSTD)
        self.ZToolbar2.addWidget(self.STD_increment)
        self.ZToolbar2.addAction(self.actionDecreaseTV)
        self.ZToolbar2.addAction(self.actionImitateHist)
        self.ZToolbar2.addAction(self.actionImitatePatchHist)
        self.ZToolbar2.addAction(self.actionFoolAdversary)
        self.ZToolbar2.addAction(self.actionIncreasePeriodicity_1D)
        self.ZToolbar2.addAction(self.actionIncreasePeriodicity)
        self.canvas.FoolAdversary_Button = self.actionFoolAdversary
        self.canvas.selectrectButton = self.selectrectButton
        self.canvas.selectpolyButton = self.selectpolyButton
        self.canvas.actionIncreasePeriodicity_1D = self.actionIncreasePeriodicity_1D
        self.canvas.actionIncreasePeriodicity = self.actionIncreasePeriodicity
        self.canvas.indicatePeriodicityButton = self.indicatePeriodicityButton
        self.canvas.periodicity_mag_1 = self.periodicity_mag_1
        self.canvas.periodicity_mag_1.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.canvas.periodicity_mag_2 = self.periodicity_mag_2
        self.canvas.periodicity_mag_2.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.STD_increment.valueChanged.connect(self.canvas.Z_optimizer_Reset)
        self.ZToolbar2.addAction(self.actionMatchSliders)
        self.ZToolbar2.addAction(self.actionProcessRandZ)
        self.ZToolbar2.addAction(self.actionProcessLimitedRandZ)
        self.Scribble_Toolbar.addWidget(self.penButton)

        self.actionFillShapes.triggered.connect(lambda s: self.canvas.set_config('fill', s))
        self.actionFillShapes.setChecked(True)
        self.open_file()
        self.show()

    def Remember_Zmap(self):
        pass#I can use this to add current Z-map to some deque to enable undo and redo
    def AutoHistTemperatureMode(self,checked):
        if checked:
            self.auto_set_hist_temperature = True
            self.canvas.Z_optimizer_Reset()
        else:
            self.auto_set_hist_temperature = False

    def DesiredHistMode(self,checked,another_image):
        self.canvas.in_picking_desired_hist_mode = 1*checked
        if checked:
            self.MasksStorage(True)
            if another_image:
                path, _ = QFileDialog.getOpenFileName(self, "Desired image for histogram imitation", "",
                                                      "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
                if path:
                    self.desired_hist_image = data_util.read_img(None, path)
                    if self.desired_hist_image.shape[2] == 3:
                        self.desired_hist_image = self.desired_hist_image[:, :, [2, 1, 0]]
                    pixmap = QPixmap()
                    pixmap.convertFromImage(qimage2ndarray.array2qimage(255*self.desired_hist_image))
                    self.canvas.setPixmap(pixmap)
                    self.canvas.setGeometry(QRect(0,0,self.desired_hist_image.shape[0],self.desired_hist_image.shape[1]))
                    self.canvas.HR_size = list(self.desired_hist_image.shape[:2])
            else:
                self.desired_hist_image = self.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0)
            self.canvas.HR_selected_mask = np.ones(self.canvas.HR_size)

        else:
            self.desired_hist_image_HR_mask = 1*self.canvas.HR_selected_mask
            self.MasksStorage(False)
            self.Update_Image_Display()
            self.actionImitateHist.setEnabled(True)
            self.actionImitatePatchHist.setEnabled(True)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()

        if self.canvas.mode == 'selectrect' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectrect_copy())

        elif self.canvas.mode == 'selectpoly' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectpoly_copy())

        else:
            clipboard.setPixmap(self.canvas.pixmap())
    def Compute_SR_Image(self):
        if self.cur_Z.size(2)==1:
            cur_Z = ((self.cur_Z * torch.ones([1, 1] + self.canvas.Z_size) - 0.5) * 2).type(self.var_L.type())
        else:
            cur_Z = self.cur_Z.type(self.var_L.type())
        self.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=cur_Z)
        self.SR_model.netG.eval()
        with torch.no_grad():
            self.SR_model.fake_H = self.SR_model.netG(self.SR_model.model_input)
            if DISPLAY_INDUCED_LR:
                self.induced_LR_image = self.SR_model.netG.module.DownscaleOP(self.SR_model.fake_H)


    def DrawRandChannel(self,min_val,max_val,uniform=False):
        return (max_val-min_val)*torch.rand([1,1]+([1,1] if uniform else self.canvas.Z_size))+min_val

    def SelectImage2Display(self,chosen_index=None):
        if chosen_index is not None:
            self.canvas.current_random_Z_index = chosen_index
        self.actionCopyFromRandom.setEnabled(self.canvas.current_random_Z_index>self.cur_Z_im_index and self.canvas.current_random_Z_index<=(self.cur_Z_im_index+self.num_random_Zs))
        if self.canvas.current_random_Z_index>=self.cur_Z_im_index:
            self.SR_model.fake_H = 1*self.canvas.random_Z_images[self.canvas.current_random_Z_index-self.cur_Z_im_index].unsqueeze(0)
            # self.cur_Z = self.canvas.random_Zs[self.canvas.current_random_Z_index-self.cur_Z_im_index,...].unsqueeze(0)
            if self.canvas.current_random_Z_index>self.cur_Z_im_index:
                self.Z_2_display = self.canvas.random_Zs[self.canvas.current_random_Z_index-self.cur_Z_im_index-1,...].unsqueeze(0)
            else:
                self.Z_2_display = self.cur_Z
        else:
            self.Z_2_display = self.no_Z_image
            if self.canvas.current_random_Z_index==self.GT_HR_index:
                self.SR_model.fake_H = 1*self.GT_HR
            elif self.canvas.current_random_Z_index==self.ESRGAN_index:
                self.SR_model.fake_H = 1*self.ESRGAN_SR

        self.Update_Image_Display(update_default_Z_image=False)

    def CopyRandom2Default(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        # self.canvas.random_Zs[0,...] = self.canvas.random_Zs[self.canvas.current_random_Z_index-self.cur_Z_im_index,...]*Z_mask+self.canvas.random_Zs[0,...]*(1-Z_mask)
        # self.cur_Z = self.canvas.random_Zs[0,...].unsqueeze(0)
        self.cur_Z = (self.canvas.random_Zs[self.canvas.current_random_Z_index-self.cur_Z_im_index-1,...].to(self.cur_Z.device)*Z_mask+self.cur_Z[0]*(1-Z_mask)).unsqueeze(0)
        self.Compute_SR_Image()
        self.canvas.random_Z_images[0] = 1*self.SR_model.fake_H[0]
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
            self.canvas.random_Z_images[i + 1] = self.SR_model.fake_H[0]
        self.cur_Z = 1*stored_Z
        self.DisplayedImageSelectionButton.setEnabled(True)
        self.SelectImage2Display()

    def Process_Random_Z(self,limited):
        if self.num_random_Zs>1 or limited:
            self.Optimize_Z('random_'+('VGG' if VGG_RANDOM_DOMAIN else 'l1')+('_limited' if limited else ''))
        else:
            UNIFORM_RANDOM = False
            Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
            if USE_SVD:
                # self.canvas.lambda0 = Z_mask*self.DrawRandChannel(0,MAX_SVD_LAMBDA,uniform=UNIFORM_RANDOM).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.lambda0
                # self.canvas.lambda1 = Z_mask*self.DrawRandChannel(0,MAX_SVD_LAMBDA,uniform=UNIFORM_RANDOM).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.lambda1
                # self.canvas.theta = Z_mask*self.DrawRandChannel(0,np.pi,uniform=UNIFORM_RANDOM).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.theta
                self.canvas.control_values = Z_mask*torch.stack([self.DrawRandChannel(0,MAX_SVD_LAMBDA,uniform=UNIFORM_RANDOM),
                    self.DrawRandChannel(0,MAX_SVD_LAMBDA,uniform=UNIFORM_RANDOM),self.DrawRandChannel(0,np.pi,uniform=UNIFORM_RANDOM)],
                    0).squeeze(0).squeeze(0)+(1-Z_mask)*self.canvas.control_values
                self.Recompose_cur_Z()
                self.canvas.Update_Z_Sliders()
            else:
                random_Z = (torch.rand([1,self.SR_model.num_latent_channels]+self.canvas.Z_size)-0.5)*2
                self.cur_Z = Z_mask*random_Z+(1-Z_mask)*self.cur_Z
            self.ReProcess()
    def Validate_Z_optimizer(self,objective):
        # if self.canvas.Z_optimizer is not None:
        if self.latest_optimizer_objective!=objective:# or objective=='hist': # Resetting optimizer in the 'patchhist' case because I use automatic tempersture search there, so I want to search each time for the best temperature.
            self.canvas.Z_optimizer_Reset()

    def MasksStorage(self,store):
        canvas_keys = ['Z_mask','HR_selected_mask','LR_mask_vertices','HR_size','random_Zs']
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
            # self.stored_lambda0 = 1*self.canvas.lambda0
            # self.stored_lambda1 = 1*self.canvas.lambda1
            # self.stored_theta = 1*self.canvas.theta
        else:
            self.canvas.control_values = 1*self.stored_control_values
            # self.canvas.lambda0 = 1*self.stored_lambda0
            # self.canvas.lambda1 = 1*self.stored_lambda1
            # self.canvas.theta = 1*self.stored_theta

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
        self.SetZ(0 if min_not_max else 1,0,reset_optimizer=False) # I'm using 1 as maximal value and not MAX_LAMBDA_VAL because I want these images to correspond to Z=[-1,1] like in the model training. Different maximal Lambda values will be manifested in the Z optimization itself when cur_Z is normalized.
        self.SetZ(0 if min_not_max else 1, 1,reset_optimizer=False)

    def Optimize_Z(self,objective):
        if NO_DC_IN_PATCH_HISTOGRAM:
            objective = objective.replace('hist', 'hist_noDC')
        if DICTIONARY_REPLACES_HISTOGRAM:
            objective = objective.replace('hist', 'dict')
        if LOCAL_STD_4_OPT:
            objective = objective.replace('STD','local_STD').replace('periodicity','local_STD_periodicity').replace('TV','local_STD_TV')
        elif L1_REPLACES_HISTOGRAM:
            objective = objective.replace('hist', 'l12GT')
        if AUTO_CYCLE_LENGTH_4_PERIODICITY:
            objective = objective.replace('periodicity', 'nonInt_periodicity')
        self.random_inits = ('random' in objective and 'limited' not in objective) or RANDOM_OPT_INITS
        self.multiple_inits = 'random' in objective or MULTIPLE_OPT_INITS
        self.Validate_Z_optimizer(objective)
        self.latest_optimizer_objective = objective
        data = {'LR':self.var_L}
        if self.canvas.Z_optimizer is None:
            # For the random_l1_limited objective, I want to have L1 differences with respect to the current non-modified image, in case I currently display another image:
            self.SR_model.fake_H = 1 * self.canvas.random_Z_images[0].unsqueeze(0)
            # self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1])
            # self.cur_Z = 1* self.canvas.random_Zs[0].unsqueeze(0)
            if not np.all(self.canvas.HR_selected_mask) and self.canvas.contained_Z_mask:#Cropping an image region to be optimized, to save on computations and allow adversarial loss
                self.optimizing_region = True
                # if np.all(self.bounding_rect[2:]<=D_EXPECTED_LR_SIZE-2*MARGINS_AROUND_REGION_OF_INTEREST) or objective=='Adversarial':
                if objective == 'Adversarial':
                    #Use this D_EXPECTED_LR_SIZE LR_image cropped size when using D that can only work with this size (non mapGAN)
                    gaps = D_EXPECTED_LR_SIZE-self.canvas.mask_bounding_rect[2:]
                    self.bounding_rect_4_opt = np.concatenate([np.maximum(self.canvas.mask_bounding_rect[:2]-gaps//2,0),np.array(2*[D_EXPECTED_LR_SIZE])])
                else:
                    self.bounding_rect_4_opt = np.concatenate([np.maximum(self.canvas.mask_bounding_rect[:2]-MARGINS_AROUND_REGION_OF_INTEREST//2,0),self.canvas.mask_bounding_rect[2:]+MARGINS_AROUND_REGION_OF_INTEREST])
                self.bounding_rect_4_opt[:2] = np.maximum([0,0],np.minimum(self.bounding_rect_4_opt[:2]+self.bounding_rect_4_opt[2:],self.canvas.LR_size[::-1])-self.bounding_rect_4_opt[2:])
                self.bounding_rect_4_opt[2:] = np.minimum(self.bounding_rect_4_opt[:2]+self.bounding_rect_4_opt[2:],self.canvas.LR_size[::-1])-self.bounding_rect_4_opt[:2]
                self.MasksStorage(True)
                self.canvas.HR_selected_mask = self.Crop2BoundingRect(self.canvas.HR_selected_mask,self.bounding_rect_4_opt,HR=True)
                # self.canvas.Z_mask = self.Crop2BoundingRect(self.canvas.Z_mask,self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
                self.canvas.Z_mask = self.Crop2BoundingRect(self.canvas.Z_mask,self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
                self.Z_mask_4_later_merging = torch.from_numpy(self.canvas.Z_mask).type(self.SR_model.fake_H.dtype).to(self.cur_Z.device)
                self.var_L = self.Crop2BoundingRect(self.var_L, self.bounding_rect_4_opt)
                data['LR'] = self.var_L
                self.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=self.Crop2BoundingRect(self.SR_model.GetLatent(),self.bounding_rect_4_opt,HR=self.canvas.HR_Z))#Because I'm saving initial Z when initializing optimizer
                self.cur_Z = self.Crop2BoundingRect(self.cur_Z,self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
                self.SR_model.fake_H = self.Crop2BoundingRect(self.SR_model.fake_H,self.bounding_rect_4_opt,HR=True) #For the limited random optimization, to have the image we want to stay close to.
                # if self.multiple_inits:
                #     self.canvas.random_Zs = self.Crop2BoundingRect(self.canvas.random_Zs,self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
            else:
                self.optimizing_region = False
                # self.bounding_rect_4_opt = copy.deepcopy(self.canvas.mask_bounding_rect)
            self.iters_per_round = 1*ITERS_PER_OPT_ROUND
            if any([phrase in objective for phrase in ['hist','dict','l12GT']]):
                # if self.desired_hist_image is None:
                #     return
                data['HR'] = torch.from_numpy(np.ascontiguousarray(np.transpose(self.desired_hist_image, (2, 0, 1)))).float().to(self.SR_model.device).unsqueeze(0)
                if 'l1' in objective and self.optimizing_region:
                    data['HR'] = self.Crop2BoundingRect(data['HR'],self.bounding_rect_4_opt,HR=True)
                data['Desired_Im_Mask'] = self.desired_hist_image_HR_mask
            elif 'desired_SVD' in objective:
                data['desired_Z'] = util.SVD_2_LatentZ(self.canvas.control_values.unsqueeze(0),max_lambda=MAX_SVD_LAMBDA)
                self.SVD_ValuesStorage(True)
                if self.optimizing_region:
                    data['desired_Z'] = self.Crop2BoundingRect(data['desired_Z'],self.bounding_rect_4_opt,HR=self.canvas.HR_Z)
                    self.canvas.control_values = self.Crop2BoundingRect(self.canvas.control_values.unsqueeze(0),self.bounding_rect_4_opt,HR=self.canvas.HR_Z).squeeze(0)
                self.Set_Extreme_SVD_Values(min_not_max=True)
                data['reference_image_min'] = 1*self.SR_model.fake_H
                self.Set_Extreme_SVD_Values(min_not_max=False)
                data['reference_image_max'] = 1*self.SR_model.fake_H
                self.SVD_ValuesStorage(False)
            elif 'periodicity' in objective:
                for p_num in range(len(self.canvas.periodicity_points)):
                    self.canvas.periodicity_points[p_num] = self.canvas.periodicity_points[p_num]*getattr(self,'periodicity_mag_%d'%(p_num+1)).value()/np.linalg.norm(self.canvas.periodicity_points[p_num])
                data['periodicity_points'] = self.canvas.periodicity_points[:2-('1D' in objective)]
            elif 'STD' in objective and not any([phrase in objective for phrase in ['periodicity','TV']]):
                data['STD_increment'] = self.STD_increment.value()
            initial_Z = 1 * self.cur_Z
            if self.multiple_inits:
                data['LR'] = data['LR'].repeat([self.num_random_Zs,1,1,1])
                # initial_Z = 1*self.canvas.random_Zs[1:]
                initial_Z = initial_Z.repeat([self.num_random_Zs,1,1,1])
                if 'random' in objective:
                    self.canvas.Z_optimizer_initial_LR = Z_OPTIMIZER_INITIAL_LR_4_RANDOM
                    if not HIGH_OPT_ITERS_LIMIT:
                        self.iters_per_round *= 3
                    if 'limited' in objective:
                        data['rmse_weight'] = 1*self.randomLimitingWeightBox.value()
            if HIGH_OPT_ITERS_LIMIT:#Signaling the optimizer to keep iterating until loss stops decreasing, using iters_per_round as window size. Still have some fixed limit.
                self.iters_per_round *= -1
            if self.canvas.Z_optimizer_logger is None:
                self.canvas.Z_optimizer_logger = []
                for i in range(self.num_random_Zs if self.multiple_inits else 1):
                    self.canvas.Z_optimizer_logger.append(Logger(self.canvas.DTE_opt,tb_logger_suffix='_%s%s'%(objective,'_%d'%(i) if self.multiple_inits else '')))
            self.canvas.Z_optimizer = util.Z_optimizer(objective=objective,Z_size=[val*self.SR_model.Z_size_factor for val in data['LR'].size()[2:]],model=self.SR_model,
                Z_range=MAX_SVD_LAMBDA,data=data,initial_LR=self.canvas.Z_optimizer_initial_LR,loggers=self.canvas.Z_optimizer_logger,max_iters=self.iters_per_round,
                image_mask=self.canvas.HR_selected_mask,Z_mask=self.canvas.Z_mask,auto_set_hist_temperature=self.auto_set_hist_temperature,
                batch_size=self.num_random_Zs if self.multiple_inits else 1,random_Z_inits=self.random_inits,initial_Z=initial_Z)
            if self.optimizing_region:
                self.MasksStorage(False)
        elif 'random' in objective:
            self.canvas.Z_optimizer.cur_iter = 0
        self.stored_Z = 1 * self.cur_Z
        if self.multiple_inits:
            self.stored_masked_zs = 1*self.canvas.random_Zs
        else:
            self.stored_masked_zs = 1 * self.cur_Z # Storing previous Z for two reasons: To recover the big picture Z when optimizing_region, and to recover previous Z if loss did not decrease
        self.cur_Z = self.canvas.Z_optimizer.optimize()
        if self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1] < 0:
            self.cur_Z = 1 * self.stored_Z
            self.SR_model.ConcatLatent(LR_image=self.var_L,latent_input=self.cur_Z.type(self.var_L.type()))
            self.SelectImage2Display()
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
                self.canvas.random_Zs = 1*self.cur_Z
                self.cur_Z = 1*self.stored_Z
                self.Process_Z_Alternatives()
                # self.cur_Z = self.canvas.random_Zs[self.canvas.current_random_Z_index,...].unsqueeze(0)
            else:
                # self.canvas.random_Zs[0] = 1*self.cur_Z.squeeze(0)
                self.DeriveControlValues()
                self.Compute_SR_Image()
                self.canvas.random_Z_images[0] = self.SR_model.fake_H[0]
                self.SelectImage2Display()
        print('%d iterations: %s loss decreased from %.2e to %.2e by %.2e (factor of %.2e)' % (len(self.canvas.Z_optimizer.loss_values), self.canvas.Z_optimizer.objective,
            self.canvas.Z_optimizer.loss_values[0],self.canvas.Z_optimizer.loss_values[-1],self.canvas.Z_optimizer.loss_values[0] - self.canvas.Z_optimizer.loss_values[-1],
            self.canvas.Z_optimizer.loss_values[-1]/self.canvas.Z_optimizer.loss_values[0]))
        if (self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))]-self.canvas.Z_optimizer.loss_values[-1])/\
                np.abs(self.canvas.Z_optimizer.loss_values[-int(np.abs(self.iters_per_round))])<1e-2*self.canvas.Z_optimizer_initial_LR: #If the loss did not decrease, I decrease the optimizer's learning rate
            self.canvas.Z_optimizer_initial_LR /= 5
            print('Loss decreased too little relative to beginning, decreasing learning rate to %.3e'%(self.canvas.Z_optimizer_initial_LR))
            self.canvas.Z_optimizer = None
        else: # This means I'm happy with this optimizer (and its learning rate), so I can cancel the auto-hist-temperature setting, in case it was set to True.
            self.auto_set_hist_temperature = False
            self.auto_hist_temperature_mode_button.setChecked(False)

    def DeriveControlValues(self):
        normalized_Z = 1*self.cur_Z.squeeze(0)
        normalized_Z[:2] = (normalized_Z[:2]+MAX_SVD_LAMBDA)/2/MAX_SVD_LAMBDA
        normalized_Z[2] /= 2
        new_control_values = torch.stack(util.SVD_Symmetric_2x2(*normalized_Z),0).to( self.canvas.control_values)# Lambda values are not guarnteed to be in [0,MAX_SVD_LAMBDA], despite Z being limited to [-MAX_SVD_LAMBDA,MAX_SVD_LAMBDA].
        self.canvas.derived_controls_indicator = np.logical_or(self.canvas.derived_controls_indicator,self.canvas.Z_mask)
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to( self.canvas.control_values)
        self.canvas.control_values = Z_mask*new_control_values+(1-Z_mask)*self.canvas.control_values
        self.canvas.Update_Z_Sliders()

    def Clear_Z_Mask(self):
        self.canvas.Z_mask = np.ones(self.canvas.Z_size)
        self.canvas.HR_selected_mask = np.ones(self.canvas.HR_size)
        self.canvas.LR_mask_vertices = [(0,0),tuple(self.canvas.LR_size[::-1])]
        self.canvas.update_mask_bounding_rect()
        self.canvas.Update_Z_Sliders()
        self.canvas.Z_optimizer_Reset()

    def Invert_Z_Mask(self):
        self.canvas.Z_mask = 1-self.canvas.Z_mask
        self.canvas.HR_selected_mask = 1-self.canvas.HR_selected_mask
        self.canvas.Z_optimizer_Reset()
        self.canvas.contained_Z_mask = not self.canvas.contained_Z_mask
        if self.canvas.contained_Z_mask:
            self.actionFoolAdversary.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.mask_bounding_rect[2:]]))
        else:
            self.actionFoolAdversary.setEnabled(np.all([val<=D_EXPECTED_LR_SIZE for val in self.canvas.HR_size]))

    def ApplyUniformZ(self):
        self.canvas.Update_Z_Sliders()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.canvas.control_values.dtype).to(self.canvas.control_values.device)
        self.canvas.control_values = Z_mask * torch.from_numpy(self.canvas.previous_sliders_values).type(Z_mask.dtype) + (1 - Z_mask) * self.canvas.control_values
        # self.cur_Z = Z_mask * torch.from_numpy(self.canvas.previous_sliders_values).type(Z_mask.dtype) + (1 - Z_mask) * self.cur_Z
        self.Recompose_cur_Z()
        self.Compute_SR_Image()
        self.Update_Image_Display()
        self.SelectImage2Display()
        self.canvas.derived_controls_indicator = (self.canvas.Z_mask*0+(1-self.canvas.Z_mask)*self.canvas.derived_controls_indicator).astype(np.bool)

        # self.canvas.random_Zs[0] = 1 * self.cur_Z.squeeze(0)

    def Recompose_cur_Z(self):
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype).to(self.cur_Z.device)
        new_Z = util.SVD_2_LatentZ(self.canvas.control_values.unsqueeze(0),max_lambda=MAX_SVD_LAMBDA).to(self.cur_Z.device)
        self.cur_Z = Z_mask * new_Z + (1 - Z_mask) * self.cur_Z
        # self.canvas.random_Zs[0] = 1*self.cur_Z.squeeze(0)

    def SetZ_And_Display(self,value,index):
        self.SetZ(value,index)
        self.Update_Image_Display()
        self.SelectImage2Display()

    def SetZ(self,value,index,reset_optimizer=True,recompose_Z=True):
        if reset_optimizer:
            self.canvas.Z_optimizer_Reset()
        Z_mask = torch.from_numpy(self.canvas.Z_mask).type(self.cur_Z.dtype)
        derived_controls_indicator = torch.from_numpy(self.canvas.derived_controls_indicator).type(self.cur_Z.dtype)
        if USE_SVD:
            # self.canvas.control_values[index] = Z_mask*value+(1-Z_mask)*self.canvas.control_values[index]
            value_increment = value - 1*self.canvas.previous_sliders_values[index]
            # value_increment = value*np.ones_like(self.canvas.previous_sliders_values[index])
            additive_values = torch.from_numpy(value_increment).type(self.canvas.control_values.dtype) + self.canvas.control_values[index]
            masked_new_values = Z_mask *(value * (1 - derived_controls_indicator) + derived_controls_indicator * additive_values)
            self.canvas.control_values[index] =  masked_new_values + (1 - Z_mask) * self.canvas.control_values[index]
            self.canvas.previous_sliders_values[index] = (1-self.canvas.Z_mask)*self.canvas.previous_sliders_values[index]+self.canvas.ReturnMaskedMapAverage(self.canvas.control_values[index].data.cpu().numpy())
            if recompose_Z:
                self.Recompose_cur_Z()
            if VERBOSITY:
                self.latent_mins = torch.min(torch.cat([self.cur_Z,self.latent_mins],0),dim=0,keepdim=True)[0]
                self.latent_maxs = torch.max(torch.cat([self.cur_Z,self.latent_maxs],0),dim=0,keepdim=True)[0]
                print(self.canvas.lambda0,self.canvas.lambda1,self.canvas.theta)
                print('mins:',[z.item() for z in self.latent_mins.view([-1])])
                print('maxs:', [z.item() for z in self.latent_maxs.view([-1])])
        else:
            raise Exception('Should recode to support Z-mask')
            self.cur_Z[0,index] = value
        if recompose_Z:
            self.Compute_SR_Image()
            if 'random_Z_images' in self.canvas.__dict__.keys():
                self.canvas.random_Z_images[0] = 1 * self.SR_model.fake_H[0]

    def ToggleDisplay_Z_Image(self):
        self.Update_Image_Display(update_default_Z_image=False)

    def Update_Image_Display(self,update_default_Z_image=True):
        pixmap = QPixmap()
        if self.ZdisplayButton.isChecked():
            im_2_display = 255/2/MAX_SVD_LAMBDA*(MAX_SVD_LAMBDA+self.Z_2_display[0].data.cpu().numpy().transpose(1,2,0)).copy()
        else:
            im_2_display = 255 * self.SR_model.fake_H.detach()[0].float().cpu().numpy().transpose(1, 2, 0).copy()
        pixmap.convertFromImage(qimage2ndarray.array2qimage(im_2_display))
        self.canvas.setPixmap(pixmap)
        if DISPLAY_INDUCED_LR:
            self.Update_LR_Display()
        if update_default_Z_image:
            self.canvas.random_Z_images[0] = 1*self.SR_model.fake_H[0]

    def Update_LR_Display(self):
        pixmap = QPixmap()
        pixmap.convertFromImage(qimage2ndarray.array2qimage(255 * self.induced_LR_image[0].data.cpu().numpy().transpose(1,2,0).copy()))
        self.LR_canvas.setPixmap(pixmap)

    def ReProcess(self):
        self.Compute_SR_Image()
        self.SelectImage2Display()
        # self.Update_Image_Display()
    #
    def Load_Z(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Z file", "",
                                              "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")
        if path:
            loaded_Z = data_util.read_img(None, path)
            loaded_Z = loaded_Z[:, :, [2, 1, 0]]
            assert list(loaded_Z.shape[:2])==self.canvas.Z_size,'Size of Z does not match image size'
            self.cur_Z = torch.from_numpy(np.transpose(2*MAX_SVD_LAMBDA*loaded_Z-MAX_SVD_LAMBDA, (2, 0, 1))).float().to(self.cur_Z.device).type(self.cur_Z.dtype).unsqueeze(0)
            self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1])
            self.Compute_SR_Image()
            self.canvas.random_Z_images[0] = self.SR_model.fake_H[0]
            self.SelectImage2Display()
            stored_mask = 1*self.canvas.Z_mask
            self.canvas.Z_mask = np.ones_like(self.canvas.Z_mask)
            self.DeriveControlValues()
            self.canvas.derived_controls_indicator = self.Estimate_DerivedControlIndicator()
            self.canvas.Z_mask = 1*stored_mask
            self.canvas.Z_optimizer_Reset()

    def Estimate_DerivedControlIndicator(self):
        PATCH_SIZE_4_ESTIMATION = 3
        patch_extraction_map = util.ReturnPatchExtractionMat(self.canvas.Z_mask,PATCH_SIZE_4_ESTIMATION,patches_overlap=1)
        STD_map = torch.sparse.mm(patch_extraction_map, self.cur_Z.mean(dim=1).view([-1, 1])).view([PATCH_SIZE_4_ESTIMATION ** 2, -1]).std(dim=0).view(
            [val-PATCH_SIZE_4_ESTIMATION+1 for val in list(self.cur_Z.size()[2:])])
        return np.pad((STD_map>0).data.cpu().numpy().astype(np.bool),pad_width=int(PATCH_SIZE_4_ESTIMATION//2),mode='edge')

    def open_file(self):
        """
        Open image file for editing, scaling the smaller dimension and cropping the remainder.
        :return:
        """
        path, _ = QFileDialog.getOpenFileName(self,"Open GT HR file" if DISPLAY_GT_HR else "Open file", "", "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")

        if path:
            loaded_image = data_util.read_img(None, path)
            if loaded_image.shape[2] == 3:
                loaded_image = loaded_image[:, :, [2, 1, 0]]
            if DISPLAY_GT_HR:
                SR_scale = self.canvas.DTE_opt['scale']
                loaded_image = loaded_image[:loaded_image.shape[0]//SR_scale*SR_scale,:loaded_image.shape[1]//SR_scale*SR_scale,:] #Removing bottom right margins to make the image shape adequate to this SR factor
                if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
                    self.GT_HR = torch.from_numpy(np.transpose(loaded_image, (2, 0, 1))).float().to(self.SR_model.device).unsqueeze(0)
                else:
                    pixmap = QPixmap()
                    pixmap.convertFromImage(qimage2ndarray.array2qimage(255 * loaded_image))
                    self.GT_canvas.setPixmap(pixmap)
                self.canvas.HR_size = list(self.GT_HR.size()[2:])
                self.var_L = self.SR_model.netG.module.DownscaleOP(torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.SR_model.device).unsqueeze(0))
            else:
                self.var_L = torch.from_numpy(np.ascontiguousarray(np.transpose(loaded_image, (2, 0, 1)))).float().to(self.SR_model.device).unsqueeze(0)
            if DISPLAY_ESRGAN_RESULTS:
                ESRGAN_opt = option.parse('./options/test/GUI_esrgan.json', is_train=False,name='RRDB_ESRGAN_x4')
                ESRGAN_opt = option.dict_to_nonedict(ESRGAN_opt)
                ESRGAN_opt['network_G']['latent_input'] = 'None'
                ESRGAN_opt['network_G']['DTE_arch'] = 0
                ESRGAN_model = create_model(ESRGAN_opt)
                ESRGAN_model.netG.eval()
                with torch.no_grad():
                    self.ESRGAN_SR = ESRGAN_model.netG(self.var_L).detach().to(torch.device('cpu'))
                self.canvas.HR_size = list(self.ESRGAN_SR.size()[2:])
                if not ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
                    pixmap = QPixmap()
                    pixmap.convertFromImage(qimage2ndarray.array2qimage(255 * self.ESRGAN_SR[0].data.cpu().numpy().transpose(1,2,0).copy()))
                    self.ESRGAN_canvas.setPixmap(pixmap)

            if 'random_Z_images' in self.canvas.__dict__.keys():
                del self.canvas.random_Z_images
            self.canvas.LR_size = list(self.var_L.size()[2:])
            self.canvas.Z_size = [val*self.canvas.DTE_opt['scale'] for val in self.canvas.LR_size] if self.canvas.HR_Z else self.canvas.LR_size
            self.canvas.Z_mask = np.ones(self.canvas.Z_size)
            self.canvas.derived_controls_indicator = np.zeros(self.canvas.Z_size)
            self.cur_Z = torch.zeros(size=[1,self.SR_model.num_latent_channels]+self.canvas.Z_size)
            self.canvas.random_Zs = self.cur_Z.repeat([self.num_random_Zs,1,1,1])
            self.canvas.previous_sliders_values = \
                np.array([self.canvas.sliderZ0.value(),self.canvas.sliderZ1.value(),self.canvas.slider_third_channel.value()]).reshape([3,1,1])/100*np.ones([1]+self.canvas.Z_size)
            # self.canvas.previous_sliders_values = np.array([0,0,0]).reshape([3,1,1])/100*np.ones([1]+self.canvas.Z_size)
            if USE_SVD:
                # self.canvas.lambda0 = torch.tensor(0.5)#*np.ones(self.canvas.LR_size)
                # self.canvas.lambda1 = torch.tensor(0.5)#*np.ones(self.canvas.LR_size)
                # self.canvas.theta = torch.tensor(0)#*np.ones(self.canvas.LR_size)
                self.canvas.control_values = 0.5*torch.ones_like(self.cur_Z).squeeze(0)
                self.SetZ(0.5*MAX_SVD_LAMBDA, 0,recompose_Z=False)  # *np.ones(self.canvas.LR_size)
                self.SetZ(0.5*MAX_SVD_LAMBDA, 1,recompose_Z=False)  # *np.ones(self.canvas.LR_size)
                self.SetZ(0.5, 2)  # *np.ones(self.canvas.LR_size)
                if VERBOSITY:
                    self.latent_mins = 100 * torch.ones([1, 3, 1, 1])
                    self.latent_maxs = -100 * torch.ones([1, 3, 1, 1])

            self.image_name = path.split('/')[-1].split('.')[0]

            self.canvas.random_Z_images = torch.cat([self.SR_model.fake_H,torch.zeros_like(self.SR_model.fake_H).repeat([self.num_random_Zs,1,1,1])],0)
            self.DisplayedImageSelectionButton.setCurrentIndex(self.cur_Z_im_index)
            if ALTERNATIVE_HR_DISPLAYS_ON_SAME_CANVAS:
                self.no_Z_image = torch.from_numpy(np.transpose(2*(resize(image=data_util.read_img(None,'./images/X.png')[:,:,::-1],output_shape=self.canvas.HR_size)-0.5), (2, 0, 1))).float().to(self.SR_model.device).unsqueeze(0)
            else:
                self.canvas.HR_size = list(self.SR_model.fake_H.size()[2:])
            self.ReProcess()
            #Scribble mask:
            pixmap = QPixmap()
            pixmap.convertFromImage(qimage2ndarray.array2qimage(np.zeros(self.canvas.HR_size)))
            self.canvas.scribble_mask.setPixmap(pixmap)

            self.canvas.setGeometry(QRect(0,0,self.canvas.HR_size[0],self.canvas.HR_size[1]))
            self.Clear_Z_Mask()

    def save_file(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "PNG Image file (*.png)")

        if path:
            imageio.imsave(path,np.clip(255*self.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0),0,255).astype(np.uint8))
            # pixmap = self.canvas.pixmap()
            # pixmap.save(path, "PNG" )

    def save_file_and_Z_map(self):
        """
        Save active canvas and cur_Z map to image file.
        :return:
        """
        # path = os.path.join('/media/ybahat/data/projects/SRGAN/GUI_outputs','%s_%d%s.png'%(self.image_name,self.saved_outputs_counter,'%s'))
        path = os.path.join('/'.join(self.canvas.DTE_opt['path']['results_root'].split('/')[:-2]),'GUI_outputs','%s_%d%s.png'%(self.image_name,self.saved_outputs_counter,'%s'))

        if path:
            imageio.imsave(path%(''),np.clip(255*self.SR_model.fake_H[0].data.cpu().numpy().transpose(1,2,0),0,255).astype(np.uint8))
            imageio.imsave(path%('_Z'),np.clip(255/2/MAX_SVD_LAMBDA*(MAX_SVD_LAMBDA+self.cur_Z[0].data.cpu().numpy().transpose(1,2,0)),0,255).astype(np.uint8))
            if DISPLAY_INDUCED_LR:
                imageio.imsave(path % ('_LR'), np.clip(255*self.induced_LR_image[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8))
            if DISPLAY_ESRGAN_RESULTS and self.saved_outputs_counter==0:
                imageio.imsave(path % ('_ESRGAN'), np.clip(255*self.ESRGAN_SR[0].data.cpu().numpy().transpose(1, 2, 0),0, 255).astype(np.uint8))
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