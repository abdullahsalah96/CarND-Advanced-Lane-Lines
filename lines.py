import cv2
import collections
import numpy as np

class Line:
    def __init__(self, buffer = 10):
        self.detected = False
        self.x_coordinates = None
        self.y_coordinates = None
        self.curvature_radius = None
        self.last_fit_pixel = None
        self.last_fit_meter = None
        self.recent_lines_pixels = collections.deque(maxlen=buffer)
        self.recent_lines_meter = collections.deque(maxlen= 2 * buffer)

    def update_line(self, new_fit_pixel, new_fit_meter, detected):
        self.detected = detected
        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_lines_pixels.append(self.last_fit_pixel)
        self.recent_lines_meter.append(self.last_fit_meter)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w = mask.shape[:2]

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_lines_pixels, axis=0)

    @property
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    def curvature_meter(self):
        y_eval = 0
        if(not self.recent_lines_meter[0] is None):
            # print(self.recent_lines_meter)
            coeffs = np.mean(self.recent_lines_meter, axis=0)
        else:
            # print("empty")
            coeffs = [-1, -1, -1]
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


