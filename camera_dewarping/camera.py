import os
from copy import deepcopy
from itertools import combinations

import click
import cv2
import numpy as np


class KeyCodes:
    NONE = 255
    EXIT = ord("q")
    SAVE = ord("s")
    PLUS = 43
    MINUS = 45
    CANCEL = ord("c")
    ESC = 27
    ENTER = 13
    PAUSE = ord(" ")
    RELEASE = ord("r")
    POP = ord("p")
    DBG_WARP_TOGGLE = ord("m")
    LOAD = ord("l")
    VISIBILITY = ord("v")


class State:
    TRIANGLE_DEFINITION = 0
    WARPING_FIX_ANCHORS = 1
    WARPING_APPLY = 2
    LOAD = 3


class Dewarping:
    def __init__(self, file):
        self.scale = 0.5
        self.video = file
        # self.videos = [
        #     "assets/left.mp4",
        #     "assets/right.mp4",
        #     "assets/center.mp4",
        # ]
        self.old_points = []
        self.points = []
        self.target_points = []
        self.old_triangles = []
        self.triangles = []
        self.cap = None
        self.selected_point = None
        self.point_threshold = 50
        self.btn_down = False
        self.warping = False
        self.circles = []
        self.lines = []
        self.old_circles = []
        self.old_lines = []
        self.state = State.TRIANGLE_DEFINITION
        self.next_state = State.TRIANGLE_DEFINITION
        self.transform_matrix = []
        self.frames = 0
        self.current_frame = 0
        self.precision = 255
        self.wait = self.precision
        # if load:
        #     self.load()

    def update_state(self):
        self.state = self.next_state

    def update_point(self, p, offset_x, offset_y, warp=False):
        points = (
            self.old_points
            if not warp
            else self.points
        )
        triangles = (
            self.old_triangles
            if not warp
            else self.triangles
        )
        new_point = (p[0] + offset_x, p[1] + offset_y)

        index = points.index(p) if p in points else None

        if index is not None:
            points[index] = new_point

        for mask in triangles:
            for i, point in enumerate(mask):
                if point[0] == p[0] and point[1] == p[1]:
                    mask[i] = new_point
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
        if self.state != State.LOAD:
            self.draw_circles(frame, warp=False)
            self.draw_lines(frame, warp=False)
            if len(self.old_circles) > 0:
                frame = self._add_layer(frame, self.old_circles)
            if len(self.old_lines) > 0:
                frame = self._add_layer(frame, self.old_lines)

            # Show warped triangle
            if self.warping:
                self.draw_circles(frame, warp=True)
                self.draw_lines(frame, warp=self.warping)
                if len(self.circles) > 0:
                    frame = self._add_layer(frame, self.circles)
                if len(self.lines) > 0:
                    frame = self._add_layer(frame, self.lines)

        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("preview", frame_downscaled)
        if self.wait > 0:
            self.wait -= 1
        else:
            cv2.setTrackbarPos("frames", "preview", self.current_frame)
            self.wait = self.precision

    def pre_warp_show(self, frame):
        background = np.zeros_like(frame)
        if self.state == State.WARPING_FIX_ANCHORS:
            if len(self.points) >= 3:
                # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
                # Consider only old points which are the ones fixed before pressing Enter
                hull = cv2.convexHull(np.array(self.old_points))
                cv2.fillConvexPoly(background, hull, (255, 255, 255))
                frame = cv2.bitwise_and(frame, background)
            else:
                frame = background
        elif self.state == State.WARPING_APPLY or self.state == State.LOAD:
            frame = self._apply_warp(frame)
        else:
            frame = background
        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("selected area", frame_downscaled)
        return frame

    def post_warp_show(self, frame):
        background = np.zeros_like(frame)
        if len(self.target_points) >= 3:
            # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
            # Consider only old points which are the ones fixed before pressing Enter
            # hull = cv2.convexHull(np.array(self.old_points))
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
            self.old_points
            if not warp
            else self.points
        )

        # check if x,y is close to any of the points
        for idx, point in enumerate(points):
            if (
                abs(point[0] - x) < self.point_threshold
                and abs(point[1] - y) < self.point_threshold
            ):
                new_point = False
                selected_point = point
                break

        if not warp:
            if new_point:
                points.append((int(x), int(y)))
                self.selected_point = points[-1]
            elif selected_point is not None:
                self.selected_point = selected_point

        # Warp
        if warp:
            if selected_point is not None and new_point is False:
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
        return frame[:, 1400:2600]


    def draw_circles(self, frame, warp=False):
        points = (
            self.old_points
            if not warp
            else self.points
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

        for point in points:  # self.old_points:
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
                self.old_triangles
                if not warp
                else self.triangles
            )
            for triangle in triangles:  # self.old_triangles:
                if warp:
                    cv2.polylines(self.lines, [triangle], True, (130, 130, 130, 255), 3)
                else:
                    cv2.polylines(self.old_lines, [triangle], True, (255, 0, 0, 255), 3)

    def get_triangles(self, frame) -> None:
        points = (
            self.old_points
            if not self.warping
            else self.points
        )
        triangles = (
            self.old_triangles
            if not self.warping
            else self.triangles
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
        for i, mask in enumerate(self.old_triangles):
            if index is not None and i != index:
                continue
            overlay = cv2.fillConvexPoly(np.zeros_like(frame), mask, (255, 255, 255))
            overlay = cv2.bitwise_and(frame, overlay)
            image_with_masks = cv2.bitwise_or(image_with_masks, overlay)

        return image_with_masks

    def on_key(self, key):
        if key == KeyCodes.EXIT:
            return False
        # elif ord("1") <= key <= ord("3"):
        #     self.current_source = key - ord("1")
        #     self.cap = cv2.VideoCapture(self.videos)

        if self.selected_point is not None:
            if key == KeyCodes.CANCEL and not self.warping:
                self.old_points.remove(self.selected_point)
                # self.points.remove(self.selected_point)
                self.selected_point = None
            elif key == KeyCodes.ESC:
                self.selected_point = None

        if key == KeyCodes.RELEASE:
            self.selected_point = None

        if key == KeyCodes.POP and not self.warping:
            if len(self.old_points) > 0:
                self.old_points.pop()

        if key == KeyCodes.ENTER:
            # self.warping = True
            if self.state == State.TRIANGLE_DEFINITION:
                self.next_state = State.WARPING_FIX_ANCHORS
                self.circles = deepcopy(
                    self.old_circles
                )
                self.lines = deepcopy(
                    self.old_lines
                )
                self.points = self.multiplicate_points(
                    self.old_triangles
                )
                self.triangles = deepcopy(
                    self.old_triangles
                )
                self.selected_point = None
                self.warping = True
                print("Warping anchors set")
            elif self.state == State.WARPING_FIX_ANCHORS:
                self.next_state = State.WARPING_APPLY
                self.save()
                print("Warping applied! ")

        if key == KeyCodes.DBG_WARP_TOGGLE:
            self.warping = not self.warping

        if key == KeyCodes.SAVE:
            self.save()

        if key == KeyCodes.LOAD and self.state != State.LOAD:
            if self.load():
                self.next_state = State.LOAD

        return True

    def save(self):
        if not os.path.exists("output"):
            os.makedirs("output")
        
        # escape slashes in path
        name = self.video.replace("/", ".")
        
        np.save(
            f"output/old_triangles_{name}.npy",
            self.old_triangles,
        )
        np.save(
            f"output/triangles_{name}.npy",
            self.triangles,
        )
        np.save(
            f"output/trans_matrix_{name}.npy",
            self.transform_matrix,
        )
        print("saved")

    def load(self):
        name = self.video.replace("/", ".")
        if os.path.exists(f"output/old_triangles_{name}.npy"):
            self.old_triangles = np.load(

                f"output/old_triangles_{name}.npy"
            )
        else:
            return False


        if os.path.exists(f"output/triangles_{name}.npy"):
            self.triangles = np.load(
                f"output/triangles_{name}.npy"
            )
        else:
            return False

        if os.path.exists(f"output/trans_matrix_{name}.npy"):
            self.transform_matrix = np.load(
                f"output/trans_matrix_{name}.npy"
            )
        else:
            return False

        self.next_state = State.LOAD
        self.update_state()
        return True

    def multiplicate_points(self, old_triangles):
        new_points = []
        for triangle in old_triangles:
            for point in triangle:
                new_points.append((point[0], point[1]))
        return new_points

    def get_triangle_masks(self, bgs):
        overlaps = []
        bgx = [(idx, bg) for idx, bg in enumerate(bgs)]
        masks = {idx: [] for idx, _ in bgx}
        for (idx1, bg1), (idx2, bg2) in list(combinations(bgx, 2)):
            # check if not intersecting
            image_intersection = cv2.bitwise_and(bg1, bg2)

            image_mask = np.zeros_like(image_intersection)
            image_mask[np.where(image_intersection >= 1)] = 255
            image_mask_inv = np.ones_like(image_mask) * 255
            image_mask_inv[np.where(image_mask >= 1)] = 0

            image_overlap = np.zeros_like(bg1)
            image_overlap = cv2.addWeighted(bg1, 0.5, bg2, 0.5, 0)

            image_common = cv2.bitwise_and(image_overlap, image_mask)

            if np.count_nonzero(image_common) > 15:
                overlaps.append(image_common)

            masks[idx1].append(image_mask_inv)
            masks[idx2].append(image_mask_inv)

        return masks, overlaps

    def blend(self, frame, bgs):
        new_frame = np.zeros_like(frame)
        if len(bgs) > 1:  # if there are more than 1 background
            masks, overlaps = self.get_triangle_masks(bgs)

            # draw non-overlapping part of the triangles
            for idx, mask in masks.items():
                triangle_crop = bgs[idx]
                for m in mask:
                    triangle_crop = cv2.bitwise_and(triangle_crop, m)

                new_frame = cv2.bitwise_or(new_frame, triangle_crop)

            if len(overlaps) == 1:  # if there is only 1 overlap, draw it
                new_frame = cv2.add(new_frame, overlaps[0])
            else:  # else get masks and draw non-overlapping part of the overlaps
                masks, over = self.get_triangle_masks(overlaps)
                for idx, mask in masks.items():
                    triangle_crop = overlaps[idx]
                    for m in mask:
                        triangle_crop = cv2.bitwise_and(triangle_crop, m)

                    new_frame = cv2.bitwise_or(new_frame, triangle_crop)

                if 0 < len(over) <= 3:  # take the smallest overlap and draw it
                    over = min(over, key=lambda x: np.count_nonzero(x))
                    new_frame = cv2.add(new_frame, over)
                elif len(over) > 3:  # otherwise blend all overlaps
                    background = np.zeros_like(frame)
                    for o in over:
                        background = cv2.add(background, o)
                    new_frame = cv2.add(new_frame, background)

        else:
            new_frame = bgs[0]

        return new_frame

    def get_recursive_masks(self, bgs):
        pass

    def _apply_warp(self, frame):
        bg = np.zeros_like(frame)

        bgs = []
        if self.state != State.LOAD:
            self.transform_matrix.clear()

        for idx, (old_triangles, triangles) in enumerate(
            zip(
                self.old_triangles,
                self.triangles,
            )
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
            if self.state == State.LOAD:
                warpMat = self.transform_matrix[idx]
            else:
                warpMat = cv2.getAffineTransform(
                    np.float32(tri1Cropped), np.float32(tri2Cropped)
                )
                self.transform_matrix.append(warpMat)
            # print(warpMat)
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

    def set_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame * self.precision)

    def render(self):
        cv2.namedWindow("preview")
        # cv2.namedWindow("intersection")
        cv2.namedWindow("selected area")
        # cv2.namedWindow("post warped")

        self.cap = cv2.VideoCapture(self.video)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) // self.precision)
        cv2.createTrackbar(
            "frames",
            "preview",
            0,
            int(self.frames),
            lambda x: self.set_frame(x),
        )

        key = KeyCodes.NONE

        while self.on_key(key):
            ret, frame = self.cap.read()
            self.current_frame = int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES) // self.precision
            )
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.setMouseCallback("preview", self.on_mouse)
            frame = self.crop(frame)
            self.get_triangles(frame)
            area = self.pre_warp_show(frame)
            # area = self.post_warp_show(frame)

            # create mask from area, where it is 0 set it to 255, else 0
            mask = np.zeros_like(area)
            mask[np.where(area == 0)] = 255
            mask[np.where(area != 0)] = 0
            # frame = np.hstack([frame, mask, area])
            # apply mask to frame
            frame = cv2.bitwise_and(frame, mask)
            frame = cv2.bitwise_or(frame, area)

            self.show(frame)

            self.update_state()
            key = cv2.waitKey(1) & 0xFF

        self.cap.release()
        cv2.destroyAllWindows()


@click.command()
@click.option("--load", default=False, is_flag=True)
@click.argument("file", type=click.Path(exists=True))
def main(file, load):
    a = Dewarping(file)
    if load:
        a.load()
    a.render()


if __name__ == "__main__":
    main()
