import cv2
import numpy as np
from copy import deepcopy

class KeyCodes():
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

class Dewarping:
    def __init__(self):
        self.current_source = 0
        self.scale = 0.5
        self.videos = [
            "assets/left.mp4",
            "assets/right.mp4",
            "assets/center.mp4",
        ]
        self.points = {0: [], 1: [], 2: []}
        self.old_points = []
        self.triangles = {0: [], 1: [], 2: []}
        self.cap = None
        self.selected_point = None
        self.movement_offset = 50
        self.point_threshold = 50
        self.btn_down = False
        self.mouse_moved = False

    def update_point(self, p, offset_x, offset_y):

        new_point = (p[0] + offset_x, p[1] + offset_y)

        index = None
        for i, point in enumerate(self.points[self.current_source]):
            if point == p:
                index = i
                break
        
        if index is not None:
            self.points[self.current_source][index] = new_point
        
        index_mask = None
        for i, mask in enumerate(self.triangles[self.current_source]):
            index_point = None
            for j, point in enumerate(mask):
                if point[0] == p[0] and point[1] == p[1]:
                    index_mask = i
                    index_point = j
                    break
            if index_mask is not None and index_point is not None:
                self.triangles[self.current_source][index_mask][index_point] = new_point

        self.selected_point = new_point
            

    def show(self, frame):
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("preview", frame_downscaled)
        key_pressed = cv2.waitKey(1)
        return key_pressed & 0xFF
    
    def on_mouse(self, event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_point = True
            selected_point = None
            self.btn_down = True
            # check if x,y is close to any of the points
            for point in self.points[self.current_source]:
                if abs(point[0] - x // self.scale) < self.point_threshold and abs(point[1] - y // self.scale) < self.point_threshold:
                    new_point = False
                    selected_point = point
                    break
            
            if new_point:
                self.points[self.current_source].append((int(x // self.scale), int(y // self.scale)))
                self.selected_point = self.points[self.current_source][-1]
            elif selected_point is not None:
                self.selected_point = selected_point


        elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
            self.mouse_moved = True
            x = max(0, min(x, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            y = max(0, min(y, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            offset_x = int(x // self.scale) - self.selected_point[0]
            offset_y = int(y // self.scale) - self.selected_point[1]


            self.update_point(self.selected_point, offset_x, offset_y)

        elif event == cv2.EVENT_LBUTTONUP and self.btn_down:
            self.btn_down = False

        elif event == cv2.EVENT_MBUTTONUP:
            self.triangle_mode = False

    
    def crop(self, frame):
        match self.current_source:
            case 0:
                return frame[:, 1400:2600]
            case 1:
                return frame[:, 1450:2550]
            case 2:
                return frame[:, 1400:2700]
    
    def get_triangles(self, frame, draw=False) -> None:
        for point in self.points[self.current_source]:
            if point == self.selected_point:
                cv2.circle(frame, point, 15, (0, 0, 255), -1)
            else:
                cv2.circle(frame, point, 15, (0, 255, 0), -1)

        if len(self.points[self.current_source]) > 3 and self.points[self.current_source] != self.old_points:

            tri = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
            # draw triangles
            for point in self.points[self.current_source]:
                tri.insert(point)

            # convert triangles to int
            triangles = tri.getTriangleList().astype(np.int32)

            self.triangles[self.current_source] = []

            # draw delaunay triangles
            for triangle in triangles:
                pt1 = tuple(triangle[0:2])
                pt2 = tuple(triangle[2:4])
                pt3 = tuple(triangle[4:6])

                self.triangles[self.current_source].append(np.array([pt1, pt2, pt3]))

            self.old_points = deepcopy(self.points[self.current_source])

        if draw:
            frame = self._draw_triangles(frame)

    def draw_masks(self, frame, index=None) -> np.ndarray:
        image_with_masks = np.zeros_like(frame)
        for i,mask in enumerate(self.triangles[self.current_source]):
            if index is not None and i != index:
                continue
            overlay = cv2.fillConvexPoly(np.zeros_like(frame), mask, (255, 255, 255))
            overlay = cv2.bitwise_and(frame, overlay)
            image_with_masks = cv2.bitwise_or(image_with_masks, overlay)

        return image_with_masks
    
    def _draw_triangles(self, frame) -> np.ndarray:
        overlay = np.zeros_like(frame)
        for triangle in self.triangles[self.current_source]:
            cv2.polylines(frame, [triangle], True, (0, 0, 0), 3)
        return frame
           
    
    def add_text(self, frame, text, x, y):
        frame = cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 12)
        frame = cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        return frame
    
    def add_info(self, frame, text):
        frame = self.add_text(frame, text, 50, 100)
        return frame

    def on_key(self, key, overlay_1, overlay_2):
        if key == KeyCodes.EXIT:
            return False
        elif key >= ord("1") and key <= ord("3"):
            self.current_source = key - ord("1")
            self.cap = cv2.VideoCapture(self.videos[self.current_source])

        if self.selected_point is not None:
            if key == KeyCodes.CANCEL:
                self.points[self.current_source].remove(self.selected_point)
                self.selected_point = None
            elif key == KeyCodes.ESC:
                self.selected_point = None
        return True

    def render(self):
        cv2.namedWindow("preview")

        self.cap = cv2.VideoCapture(self.videos[self.current_source])
        key, overlay_1, overlay_2 = -1, None, None

        while self.on_key(key, overlay_1, overlay_2):
            ret, frame = self.cap.read()
            if not ret:
                # restart video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.setMouseCallback("preview", self.on_mouse)
            frame = self.crop(frame)
            self.get_triangles(frame, draw=True)

            # overlay = self.draw_masks(frame)
            # full = np.hstack((frame, overlay))
            key = self.show(frame)
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    a = Dewarping()
    a.render()