import cv2
import numpy as np
import random

class Params():
    NONE = 255
    EXIT = ord("q")

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
        self.masks = {0: [], 1: [], 2: []}
        self.transforms = {0: [], 1: [], 2: []}
        self.last_points = {0: 0, 1: 0, 2: 0}
        self.cap = None

    def show(self, frame):
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("preview", frame_downscaled)
        return cv2.waitKey(1) & 0xFF
    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.current_source].append((int(x // self.scale), int(y // self.scale)))
    
    def crop(self, frame):
        match self.current_source:
            case 0:
                return frame[:, 1400:2600]
            case 1:
                return frame[:, 1450:2550]
            case 2:
                return frame[:, 1400:2700]
    
    def get_triangles(self, frame):
        if len(self.points[self.current_source]) > 3:

            tri = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
            # draw triangles
            for point in self.points[self.current_source]:
                tri.insert(point)

            # convert triangles to int
            triangles = tri.getTriangleList().astype(np.int32)

            self.masks[self.current_source] = []

            # draw delaunay triangles
            for triangle in triangles:
                pt1 = tuple(triangle[0:2])
                pt2 = tuple(triangle[2:4])
                pt3 = tuple(triangle[4:6])
                # pt1 = (pt1[0] + random.randint(-200, 200), pt1[1])
                # pt2 = (pt2[0] + random.randint(-200, 200), pt2[1])
                # pt3 = (pt3[0] + random.randint(-200, 200), pt3[1])

                self.masks[self.current_source].append(np.array([pt1, pt2, pt3]))

    def draw_masks(self, frame, index=None):
        image_with_masks = np.zeros_like(frame)
        for i,mask in enumerate(self.masks[self.current_source]):
            if index is not None and i != index:
                continue
            overlay = cv2.fillConvexPoly(np.zeros_like(frame), mask, (255, 255, 255))
            overlay = cv2.bitwise_and(frame, overlay)
            image_with_masks = cv2.bitwise_or(image_with_masks, overlay)

        return image_with_masks

    def on_key(self, key, overlay_1, overlay_2):
        if key == Params.EXIT:
            return False
        elif key >= ord("1") and key <= ord("3"):
            self.current_source = key - ord("1")
            self.cap = cv2.VideoCapture(self.videos[self.current_source])
        elif key == ord("s"):
            cv2.imwrite("output/0.jpg", overlay_1)
            cv2.imwrite("output/1.jpg", overlay_2)
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
            self.get_triangles(frame)

            overlay_1 = self.draw_masks(frame, 0)
            overlay_2 = self.draw_masks(frame, 1)
            full = np.hstack((frame, overlay_1, overlay_2))
            key = self.show(full)
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    a = Dewarping()
    a.render()