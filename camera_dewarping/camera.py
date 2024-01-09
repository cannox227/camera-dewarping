from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial import ConvexHull

class KeyCodes:
    NONE = 255
    EXIT = ord("q")
    LEFT = ord("a")
    RIGHT = ord("d")
    UP = ord("w")
    DOWN = ord("s")
    PLUS = 43
    MINUS = 45
    CANCEL = ord("c")
    ESC = 27
    ENTER = 13
    PAUSE = ord(" ")
    RELEASE = ord("r")
    POP = ord("p")


class Dewarping:
    def __init__(self):
        self.current_source = 0
        self.scale = 0.5
        self.videos = [
            "assets/left.mp4",
            "assets/right.mp4",
            "assets/center.mp4",
        ]
        self.old_points = {0: [], 1: [], 2: []}
        self.points = {0: [], 1: [], 2: []}
        self.old_triangles = {0: [], 1: [], 2: []}
        self.triangles = {0: [], 1: [], 2: []}
        self.cap = None
        self.selected_point = None
        self.point_threshold = 50
        self.btn_down = False
        self.warping = False
        self.background = None
        self.overlay = None
        self.circles = None
        self.lines = None
        self.old_circles = None
        self.old_lines = None

    def update_point(self, p, offset_x, offset_y, warp=False):
        points = self.old_points[self.current_source] if not warp else self.points[self.current_source]
        triangles = self.old_triangles[self.current_source] if not warp else self.triangles[self.current_source]
        new_point = (p[0] + offset_x, p[1] + offset_y)

        index = None
        for i, point in enumerate(points):
            if point == p:
                index = i
                break

        if index is not None:
            points[index] = new_point

        index_mask = None
        for i, mask in enumerate(triangles):
            index_point = None
            for j, point in enumerate(mask):
                if point[0] == p[0] and point[1] == p[1]:
                    index_mask = i
                    index_point = j
                    break
            if index_mask is not None and index_point is not None:
                triangles[index_mask][index_point] = new_point

        self.selected_point = new_point

    def _add_layer(self, frame, layer):
        if frame.shape != layer.shape:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        mask = 255 - layer[:, :, 3]
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = cv2.bitwise_or(frame, layer)
        return frame

    def show(self, frame):
        self.draw_circles(frame, warp=False)
        self.draw_lines(frame, warp=False)
        if self.old_circles is not None:
            frame = self._add_layer(frame, self.old_circles)
        if self.old_lines is not None:
            frame = self._add_layer(frame, self.old_lines)

        if self.warping:
            self.draw_circles(frame, warp=True)
            # self.draw_lines(frame, warp=self.warping)
            if self.circles is not None:
                frame = self._add_layer(frame, self.circles)
            # if self.lines is not None:
                # frame = self._add_layer(frame, self.lines)
        
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("preview", frame_downscaled)
        key_pressed = cv2.waitKey(1)
        return key_pressed & 0xFF
    
    def warp_show(self, frame):
        background = np.zeros_like(frame)
        if len(self.points[self.current_source]) >= 3:
            # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
            hull = cv2.convexHull(np.array(self.points[self.current_source]))
            cv2.fillConvexPoly(background, hull, (255, 255, 255))
            frame = cv2.bitwise_and(frame, background)
        else:
            frame = background
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("warped", frame_downscaled)
        key_pressed = cv2.waitKey(1)
        return key_pressed & 0xFF 

    def _select_point(self, x, y, warp=False):
        new_point = True
        selected_point = None
        self.btn_down = True
        points = self.old_points[self.current_source] if not warp else self.points[self.current_source]

        # check if x,y is close to any of the points
        for point in points:
            if (
                abs(point[0] - x) < self.point_threshold
                and abs(point[1] - y) < self.point_threshold
            ):
                new_point = False
                selected_point = point
                break
        
        if new_point:
            points.append((int(x), int(y)))
            self.selected_point = points[-1]
        elif selected_point is not None:
            self.selected_point = selected_point



    def on_mouse(self, event, x, y, flags, data):
        x = x // self.scale
        y = y // self.scale

        if not self.warping:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._select_point(x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
                offset_x = int(x) - self.selected_point[0]
                offset_y = int(y) - self.selected_point[1]

                self.update_point(self.selected_point, offset_x, offset_y)

            elif event == cv2.EVENT_LBUTTONUP and self.btn_down:
                self.btn_down = False
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._select_point(x, y, warp=True)

            elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
                offset_x = int(x) - self.selected_point[0]
                offset_y = int(y) - self.selected_point[1]

                self.update_point(self.selected_point, offset_x, offset_y, warp=True)
            
            elif event == cv2.EVENT_LBUTTONUP and self.btn_down:
                self.btn_down = False

    def crop(self, frame):
        match self.current_source:
            case 0:
                return frame[:, 1400:2600]
            case 1:
                return frame[:, 1450:2550]
            case 2:
                return frame[:, 1400:2700]

    def draw_circles(self, frame, warp=False):

        points = self.old_points[self.current_source] if not warp else self.points[self.current_source]

        if points != []:
            if warp:
                self.circles = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )
            else:
                self.old_circles = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )


        for point in self.old_points[self.current_source]:
            # Colors are in BGRA format
            if warp:
                if point == self.selected_point:
                    cv2.circle(self.circles, point, 15, (255, 0, 255, 255), -1)
                else:
                    cv2.circle(self.circles, point, 15, (0, 255, 255, 255), -1)
            
            else:
                if point == self.selected_point:
                    cv2.circle(self.old_circles, point, 15, (0, 0, 255, 255), -1)
                else:
                    cv2.circle(self.old_circles, point, 15, (0, 255, 0, 255), -1)

    def draw_lines(self, frame, warp=False):
        if self.old_triangles != []:
            self.old_lines = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
            )
            for triangle in self.old_triangles[self.current_source]:
                if warp:
                    cv2.polylines(self.lines, [triangle], True, (255, 255, 0, 255), 3)
                else:
                    cv2.polylines(self.old_lines, [triangle], True, (255, 0, 0, 255), 3)

    def get_triangles(self, frame, draw=False) -> None:
        if len(self.old_points[self.current_source]) > 2:
            tri = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
            # draw triangles
            for point in self.old_points[self.current_source]:
                tri.insert(point)

            # convert triangles to int
            triangles = tri.getTriangleList().astype(np.int32)

            self.old_triangles[self.current_source] = []

            # draw delaunay triangles
            for triangle in triangles:
                pt1 = tuple(triangle[0:2])
                pt2 = tuple(triangle[2:4])
                pt3 = tuple(triangle[4:6])

                self.old_triangles[self.current_source].append(np.array([pt1, pt2, pt3]))

    def draw_masks(self, frame, index=None) -> np.ndarray:
        image_with_masks = np.zeros_like(frame)
        for i, mask in enumerate(self.old_triangles[self.current_source]):
            if index is not None and i != index:
                continue
            overlay = cv2.fillConvexPoly(np.zeros_like(frame), mask, (255, 255, 255))
            overlay = cv2.bitwise_and(frame, overlay)
            image_with_masks = cv2.bitwise_or(image_with_masks, overlay)

        return image_with_masks

    def _draw_triangles(self, frame) -> None:
        # create transparent overlay (4 channels)
        self.overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        # draw triangles
        for triangle in self.old_triangles[self.current_source]:
            cv2.polylines(self.overlay, [triangle], True, (0, 0, 255, 128), 3)
            # cv2.fillConvexPoly(self.overlay, triangle, (255, 255, 255, 255))

    def warp(self, frame):
        if self.warped_points == []:
            self.background = deepcopy(frame)
            for triangle in self.old_triangles[self.current_source]:
                pt1, pt2, pt3 = triangle
                self.warped_points.append(np.array([pt1, pt2, pt3]))
        #
        background = deepcopy(self.background)
        for triangle in self.warped_points:
            for vertex in triangle:
                cv2.circle(background, tuple(vertex), 15, (255, 0, 0), -1)
            cv2.polylines(background, [triangle], True, (255, 255, 255), 3)

        return background

    def on_key(self, key):
        if key == KeyCodes.EXIT:
            return False
        elif ord("1") <= key <= ord("3"):
            self.current_source = key - ord("1")
            self.cap = cv2.VideoCapture(self.videos[self.current_source])

        if self.selected_point is not None:
            if key == KeyCodes.CANCEL:
                self.old_points[self.current_source].remove(self.selected_point)
                # self.points[self.current_source].remove(self.selected_point)
                self.selected_point = None
            elif key == KeyCodes.ESC:
                self.selected_point = None
        
        if key == KeyCodes.RELEASE:
            self.selected_point = None
        
        if key == KeyCodes.POP:
            if len(self.old_points[self.current_source]) > 0:
                self.old_points[self.current_source].pop()


        if key == KeyCodes.ENTER:
            self.warping = True
            self.circles = deepcopy(self.old_circles)
            self.lines = deepcopy(self.old_lines)
            self.points = deepcopy(self.old_points)
            self.triangles = deepcopy(self.old_triangles)
            self.selected_point = None
        return True

    def point_remove_check(self, points):
        if len(points) > 3:
            return True

    def render(self):
        cv2.namedWindow("preview")

        cv2.namedWindow("warped")
        self.cap = cv2.VideoCapture(self.videos[self.current_source])
        key = KeyCodes.NONE
        
        while self.on_key(key):
            ret, frame = self.cap.read()
            self.background = frame
            if not ret:
                # restart video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.setMouseCallback("preview", self.on_mouse)
            frame = self.crop(frame)
            self.get_triangles(frame)
            key = self.show(frame)
            key = self.warp_show(frame)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    a = Dewarping()
    a.render()
