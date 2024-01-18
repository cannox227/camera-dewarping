from copy import deepcopy
from itertools import combinations

import cv2
import numpy as np


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
    DBG_WARP_TOGGLE = ord("m")


class State:
    TRIANGLE_DEFINITION = 0
    WARPING_FIX_ANCHORS = 1
    WARPING_APPLY = 2


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
        self.target_points = {0: [], 1: [], 2: []}
        self.old_triangles = {0: [], 1: [], 2: []}
        self.target_triangles = {0: [], 1: [], 2: []}
        self.triangles = {0: [], 1: [], 2: []}
        self.cap = None
        self.selected_point = None
        self.selected_point_warp_index = None
        self.point_threshold = 25
        self.btn_down = False
        self.warping = False
        self.background = None
        self.overlay = None
        self.circles = {0: [], 1: [], 2: []}
        self.lines = {0: [], 1: [], 2: []}
        self.old_circles = {0: [], 1: [], 2: []}
        self.old_lines = {0: [], 1: [], 2: []}
        self.state = State.TRIANGLE_DEFINITION
        self.next_state = State.TRIANGLE_DEFINITION

    def update_state(self):
        self.state = self.next_state

    def update_point(self, p, offset_x, offset_y, warp=False):
        points = (
            self.old_points[self.current_source]
            if not warp
            else self.points[self.current_source]
        )
        triangles = (
            self.old_triangles[self.current_source]
            if not warp
            else self.triangles[self.current_source]
        )
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
        if frame.shape != layer.shape or frame.shape[2] != 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            if layer.shape[2] != 4:
                layer = cv2.cvtColor(layer, cv2.COLOR_BGR2BGRA)

        mask = 255 - layer[:, :, 3]
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = cv2.bitwise_or(frame, layer)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def show(self, frame):
        # Show both old triangle and new triangle (to be warped)
        self.draw_circles(frame, warp=False)
        self.draw_lines(frame, warp=False)
        if len(self.old_circles[self.current_source]) > 0:
            frame = self._add_layer(frame, self.old_circles)
        if len(self.old_lines[self.current_source]) > 0:
            frame = self._add_layer(frame, self.old_lines)

        # Show warped triangle
        if self.warping:
            self.draw_circles(frame, warp=True)
            self.draw_lines(frame, warp=self.warping)
            if len(self.circles[self.current_source]) > 0:
                frame = self._add_layer(frame, self.circles)
            if len(self.lines[self.current_source]) > 0:
                frame = self._add_layer(frame, self.lines)

        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("preview", frame_downscaled)

    def pre_warp_show(self, frame):
        background = np.zeros_like(frame)
        if self.state == State.WARPING_APPLY:
            if len(self.points[self.current_source]) >= 3:
                # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
                # Consider only old points which are the ones fixed before pressing Enter
                hull = cv2.convexHull(np.array(self.points[self.current_source]))
                cv2.fillConvexPoly(background, hull, (255, 255, 255))
                frame = cv2.bitwise_and(frame, background)
                frame = self._apply_warp(frame)
            else:
                frame = background
            # elif self.state == State.WARPING_APPLY:
            # frame = self._apply_warp(frame)
        else:
            frame = background
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("selected area", frame_downscaled)
        return frame

    def post_warp_show(self, frame):
        background = np.zeros_like(frame)
        if len(self.target_points[self.current_source]) >= 3:
            # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
            # Consider only old points which are the ones fixed before pressing Enter
            # hull = cv2.convexHull(np.array(self.old_points[self.current_source]))
            # cv2.fillConvexPoly(background, hull, (255, 255, 255))
            # frame = cv2.bitwise_and(frame, background)
            pass
        else:
            frame = background
        # frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        # cv2.imshow("post warped", frame_downscaled)
        return frame

    def _select_point(self, x, y, warp=False):
        new_point = True
        selected_point = None
        selected_point_index = None
        self.btn_down = True
        points = (
            self.old_points[self.current_source]
            if not warp
            else self.points[self.current_source]
        )

        # check if x,y is close to any of the points
        for idx, point in enumerate(points):
            if (
                abs(point[0] - x) < self.point_threshold
                and abs(point[1] - y) < self.point_threshold
            ):
                new_point = False
                selected_point = point
                selected_point_index = idx
                print(
                    f"Point selected {(x,y)} found in list wrt to point: {point} at index {selected_point_index}"
                )
                break

        if not warp:
            if new_point:
                points.append((int(x), int(y)))
                self.selected_point = points[-1]
            elif selected_point is not None:
                self.selected_point = selected_point

        # Warp
        if warp:
            if selected_point is not None and new_point == False:
                self.selected_point = selected_point
                self.selected_point_warp_index = selected_point_index

    def on_mouse(self, event, x, y, flags, data):
        x = x // self.scale
        y = y // self.scale

        # clip x,y to frame size
        x = max(0, min(x, self.background.shape[1] - 1))
        y = max(0, min(y, self.background.shape[0] - 1))

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
        points = (
            self.old_points[self.current_source]
            if not warp
            else self.points[self.current_source]
        )

        if points != []:
            if warp:
                self.circles = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )
            else:
                self.old_circles = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )

        for point in points:  # self.old_points[self.current_source]:
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
            self.lines = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            triangles = (
                self.old_triangles[self.current_source]
                if not warp
                else self.triangles[self.current_source]
            )
            for triangle in triangles:  # self.old_triangles[self.current_source]:
                if warp:
                    cv2.polylines(self.lines, [triangle], True, (130, 130, 130, 255), 3)
                else:
                    cv2.polylines(self.old_lines, [triangle], True, (255, 0, 0, 255), 3)

    def get_triangles(self, frame, draw=False) -> None:
        points = (
            self.old_points[self.current_source]
            if not self.warping
            else self.points[self.current_source]
        )
        triangles = (
            self.old_triangles[self.current_source]
            if not self.warping
            else self.triangles[self.current_source]
        )

        if State.TRIANGLE_DEFINITION == self.state and not self.warping:
            if len(points) > 2:
                tri = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
                # draw triangles
                for point in points:
                    tri.insert(point)

                # convert triangles to int
                cv2_triangles = tri.getTriangleList().astype(np.int32)

                triangles.clear()

                # draw delaunay triangles
                for triangle in cv2_triangles:
                    pt1 = tuple(triangle[0:2])
                    pt2 = tuple(triangle[2:4])
                    pt3 = tuple(triangle[4:6])

                    triangles.append(np.array([pt1, pt2, pt3]))

        else:
            if len(points) > 2:
                triangles.clear()
                for i in range(0, len(points), 3):
                    triangles.append(
                        np.array([points[i], points[i + 1], points[i + 2]])
                    )

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
            if key == KeyCodes.CANCEL and not self.warping:
                self.old_points[self.current_source].remove(self.selected_point)
                # self.points[self.current_source].remove(self.selected_point)
                self.selected_point = None
            elif key == KeyCodes.ESC:
                self.selected_point = None

        if key == KeyCodes.RELEASE:
            self.selected_point = None
            self.selected_point_warp_index = None

        if key == KeyCodes.POP and not self.warping:
            if len(self.old_points[self.current_source]) > 0:
                self.old_points[self.current_source].pop()

        if key == KeyCodes.ENTER:
            # self.warping = True
            if self.state == State.TRIANGLE_DEFINITION:
                self.next_state = State.WARPING_APPLY
                self.circles[self.current_source] = deepcopy(
                    self.old_circles[self.current_source]
                )
                self.lines[self.current_source] = deepcopy(
                    self.old_lines[self.current_source]
                )
                self.points[self.current_source] = self.multiplicate_points(
                    self.old_triangles[self.current_source]
                )
                self.triangles[self.current_source] = deepcopy(
                    self.old_triangles[self.current_source]
                )
                self.selected_point = None
                self.warping = True
                print("Warping anchors set")
            elif self.state == State.WARPING_FIX_ANCHORS:
                self.next_state = State.WARPING_APPLY
                print("Warping applied! ")

        if key == KeyCodes.DBG_WARP_TOGGLE:
            self.warping = not self.warping
        return True

    def multiplicate_points(self, old_triangles):
        new_points = []
        for triangle in old_triangles:
            for point in triangle:
                new_points.append((point[0], point[1]))
        return new_points

    def point_remove_check(self, points):
        if len(points) > 3:
            return True

    def blend(self, frame, bgs):
        new_frame = np.zeros_like(frame)
        print(len(bgs))
        if len(bgs) > 1:
            for bg1, bg2 in list(combinations(bgs, 2)):

                # bg1 = cv2.GaussianBlur(bg1, (5, 5), 0)
                # bg2 = cv2.GaussianBlur(bg2, (5, 5), 0)

                image_intersection = cv2.bitwise_and(bg1, bg2)
                image_mask = np.zeros_like(image_intersection)
                image_mask[np.where(image_intersection >= 1)] = 255
                image_mask_inv = np.ones_like(image_mask) * 255
                image_mask_inv[np.where(image_mask >= 1)] = 0

                image_overlap = np.zeros_like(bg1)
                image_overlap = cv2.addWeighted(bg1, 0.5, bg2, 0.5, 0)

                image_common = cv2.bitwise_and(image_overlap, image_mask)

                image_final = image_common.copy()

                bg1_new = cv2.bitwise_and(bg1, image_mask_inv)
                bg2_new = cv2.bitwise_and(bg2, image_mask_inv)

                image_final = cv2.addWeighted(image_final, 1, bg1_new, 1, 0)
                image_final = cv2.addWeighted(image_final, 1, bg2_new, 1, 0)
                new_frame = image_final

                new_frame = cv2.GaussianBlur(new_frame, (5, 5), 0)

        else:
            new_frame = bgs[0]

        return new_frame

    def _apply_warp(self, frame):
        bg = np.zeros_like(frame)

        bgs = []

        for old_triangles, triangles in zip(
            self.old_triangles[self.current_source], self.triangles[self.current_source]
        ):
            tri1 = np.float32(old_triangles)
            tri2 = np.float32(triangles)

            current_bg = np.zeros_like(frame)

            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)
            mask = np.zeros_like(frame, dtype=np.uint8)
            tri1Cropped = []
            tri2Cropped = []

            for i in range(0, 3):
                tri1Cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
                tri2Cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

            # Crop input image
            img1Cropped = frame[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
            warpMat = cv2.getAffineTransform(
                np.float32(tri1Cropped), np.float32(tri2Cropped)
            )
            img2Cropped = cv2.warpAffine(
                img1Cropped,
                warpMat,
                (r2[2], r2[3]),
                None,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

            img2Cropped = img2Cropped * mask

            # Copy triangular region of the rectangular patch to the output image
            current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = bg[
                r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
            ] * ((1.0, 1.0, 1.0) - mask)

            current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
                bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Cropped
            )

            bgs.append(current_bg)

        bg = self.blend(frame, bgs)
        return bg
        # transform_matrix = cv2.getAffineTransform(
        #     tri1, tri2)
        # return cv2.warpAffine(frame, transform_matrix, (frame.shape[1], frame.shape[0]))
        # return cv2.warpAffine(frame, transform_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    def render(self):
        cv2.namedWindow("preview")

        cv2.namedWindow("selected area")
        # cv2.namedWindow("post warped")

        self.cap = cv2.VideoCapture(self.videos[self.current_source])
        key = KeyCodes.NONE

        while self.on_key(key):
            ret, frame = self.cap.read()
            if not ret:
                # restart video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = self.crop(frame)
            cv2.setMouseCallback("preview", self.on_mouse)
            self.get_triangles(frame)
            if self.state != State.TRIANGLE_DEFINITION:
                area = self.pre_warp_show(frame)
                print(self.points)
                print(self.triangles)
                # area = self.post_warp_show(frame)

                # create mask from area, where it is 0 set it to 255, else 0
                mask = np.zeros_like(area)
                mask[np.where(area == 0)] = 255
                mask[np.where(area != 0)] = 0
                # frame = np.hstack([frame, mask, area])
                # apply mask to frame
                frame = cv2.bitwise_and(frame, mask)
                frame = cv2.bitwise_or(frame, area)

            self.background = frame
            self.show(frame)

            self.update_state()
            key = cv2.waitKey(1) & 0xFF

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    a = Dewarping()
    a.render()
