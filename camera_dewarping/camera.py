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
    CANCEL = ord("c")
    ENTER = 13
    RELEASE = ord("r")
    POP = ord("p")
    DBG_WARP_TOGGLE = ord("m")
    LOAD = ord("l")
    VISIBILITY = ord("v")
    GROUP = ord("g")
    DRAW = ord("d")


class State:
    TRIANGLE_DEFINITION = 0
    WARPING_FIX_ANCHORS = 1
    WARPING_APPLY = 2
    LOAD = 3


class Dewarping:
    def __init__(self, file, load, scale):
        self.scale = scale
        self.video = file
        self.old_points = [[]]
        self.points = [[]]
        self.target_points = [[]]
        self.old_triangles = [[]]
        self.triangles = [[]]
        self.cap = None
        self.selected_point = None
        self.point_threshold = 50
        self.btn_down = False
        self.warping = False
        self.circles = [[]]
        self.lines = [[]]
        self.old_circles = [[]]
        self.old_lines = [[]]
        self.state = State.TRIANGLE_DEFINITION
        self.next_state = State.TRIANGLE_DEFINITION
        self.transform_matrix = [[]]
        self.frames = 0
        self.current_frame = 0
        self.precision = 255
        self.wait = self.precision
        self.group = 0
        self.draw = True
        self.visibility = True
        if load:
            if not self.load():
                print("No saved data found, skipping")

    def update_state(self):
        self.state = self.next_state

    def update_point(self, p, offset_x, offset_y, warp=False):
        for idx in range(self.group + 1):
            points = self.old_points[idx] if not warp else self.points[idx]
            triangles = self.old_triangles[idx] if not warp else self.triangles[idx]
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
            for idx in range(self.group + 1):
                if self.draw:
                    self.draw_circles(frame, warp=False)
                    self.draw_lines(frame, warp=False)
                    if len(self.old_circles[idx]) > 0:
                        frame = self._add_layer(frame, self.old_circles[idx])
                    if len(self.old_lines[idx]) > 0:
                        frame = self._add_layer(frame, self.old_lines[idx])

                    # Show warped triangle
                    if self.warping:
                        self.draw_circles(frame, warp=True)
                        self.draw_lines(frame, warp=self.warping)
                        if len(self.circles[idx]) > 0:
                            frame = self._add_layer(frame, self.circles[idx])
                        if len(self.lines[idx]) > 0:
                            frame = self._add_layer(frame, self.lines[idx])

        frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        cv2.imshow("Camera Dewarping", frame_downscaled)
        if self.wait > 0:
            self.wait -= 1
        else:
            cv2.setTrackbarPos("frames", "Camera Dewarping", self.current_frame)
            self.wait = self.precision

    def pre_warp_show(self, frame):
        background = np.zeros_like(frame)
        if self.state == State.WARPING_FIX_ANCHORS:
            hulls = []
            for idx in range(self.group + 1):
                temp_background = np.zeros_like(frame)
                if len(self.points[idx]) >= 3:
                    # get convex hull and apply mask (frame = cv2.bitwise_and(frame, background, mask=background))
                    # Consider only old points which are the ones fixed before pressing Enter
                    hull = cv2.convexHull(np.array(self.old_points[idx]))
                    cv2.fillConvexPoly(temp_background, hull, (255, 255, 255))
                    # frame = cv2.bitwise_and(frame, background)
                    hulls.append(temp_background)
                else:
                    frame = background

            for hull in hulls:
                background = cv2.bitwise_or(background, hull)

            frame = cv2.bitwise_and(frame, background)

        elif self.state == State.WARPING_APPLY or self.state == State.LOAD:
            frame = self._apply_warp(frame)
        else:
            frame = background
        # frame_downscaled = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        # cv2.imshow("selected area", frame_downscaled)
        return frame

    def _select_point(self, x, y, warp=False):
        new_point = True
        selected_point = None
        self.btn_down = True

        for idx in range(self.group + 1):
            points = self.old_points[idx] if not warp else self.points[idx]

            # check if x,y is close to any of the points
            for point in points:
                if (
                    abs(point[0] - x) < self.point_threshold
                    and abs(point[1] - y) < self.point_threshold
                ):
                    new_point = False
                    selected_point = point
                    break

        if not warp:
            if new_point:
                points = (
                    self.old_points[self.group] if not warp else self.points[self.group]
                )
                points.append((int(x), int(y)))
                self.selected_point = points[-1]
            elif selected_point is not None:
                self.selected_point = selected_point
        else:
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

    def draw_circles(self, frame, warp=False):
        for idx in range(self.group + 1):
            points = self.old_points[idx] if not warp else self.points[idx]

            if points != []:
                if warp:
                    self.circles[idx] = np.zeros(
                        (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                    )
                else:
                    self.old_circles[idx] = np.zeros(
                        (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                    )

            for point in points:  # self.old_points:
                # Colors are in BGRA format
                if warp:
                    if point == self.selected_point:
                        cv2.circle(self.circles[idx], point, 15, (255, 0, 255, 255), -1)
                    else:
                        cv2.circle(self.circles[idx], point, 15, (0, 255, 255, 255), -1)

                else:
                    if point == self.selected_point:
                        cv2.circle(
                            self.old_circles[idx], point, 15, (0, 0, 255, 255), -1
                        )
                    else:
                        cv2.circle(
                            self.old_circles[idx], point, 15, (0, 255, 0, 255), -1
                        )

    def draw_lines(self, frame, warp=False):
        for idx in range(self.group + 1):
            if self.old_triangles[idx]:
                self.old_lines[idx] = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )
                self.lines[idx] = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )
                triangles = self.old_triangles[idx] if not warp else self.triangles[idx]
                print(triangles)
                for triangle in triangles:  # self.old_triangles:
                    if warp:
                        cv2.polylines(
                            self.lines[idx], [triangle], True, (130, 130, 130, 255), 3
                        )
                    else:
                        cv2.polylines(
                            self.old_lines[idx], [triangle], True, (255, 0, 0, 255), 3
                        )

    def get_triangles(self, frame) -> None:
        for idx in range(self.group + 1):
            points = self.old_points[idx] if not self.warping else self.points[idx]
            triangles = (
                self.old_triangles[idx] if not self.warping else self.triangles[idx]
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
                    self.old_triangles[idx].clear()
                    self.old_lines[idx] = np.zeros(
                        (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                    )

            else:
                if len(points) > 2:
                    triangles.clear()
                    for i in range(0, len(points), 3):
                        triangles.append(
                            np.array([points[i], points[i + 1], points[i + 2]])
                        )

    def on_key(self, key):
        if key == KeyCodes.EXIT:
            return False

        if self.selected_point is not None:
            if key == KeyCodes.CANCEL and not self.warping:
                for idx in range(self.group + 1):
                    if self.selected_point in self.old_points[idx]:
                        self.old_points[idx].remove(self.selected_point)
                        if len(self.old_points[idx]) == 0:
                            self.old_triangles[idx].clear()
                            self.group -= 1
                # self.points.remove(self.selected_point)
                self.selected_point = None

        if key == KeyCodes.RELEASE:
            self.selected_point = None

        if key == KeyCodes.POP and not self.warping:
            idx = self.group
            if len(self.old_points[idx]) > 0:
                self.old_points[idx].pop()
                if len(self.old_points[idx]) == 0:
                    self.old_triangles[idx].clear()
                    self.group -= 1

        if key == KeyCodes.ENTER:
            # self.warping = True
            if self.state == State.TRIANGLE_DEFINITION:
                self.next_state = State.WARPING_FIX_ANCHORS
                for idx in range(self.group + 1):
                    self.circles[idx] = deepcopy(self.old_circles[idx])
                    self.lines[idx] = deepcopy(self.old_lines[idx])
                    self.points[idx] = self.multiplicate_points(self.old_triangles[idx])
                    self.triangles[idx] = deepcopy(self.old_triangles[idx])
                self.selected_point = None
                self.warping = True
                print("Warping anchors set")
            elif self.state == State.WARPING_FIX_ANCHORS:
                self.next_state = State.WARPING_APPLY
                print("Warping applied!")

        if key == KeyCodes.DBG_WARP_TOGGLE:
            self.warping = not self.warping

        if key == KeyCodes.SAVE:
            self.save()

        if key == KeyCodes.LOAD and self.state != State.LOAD:
            if self.load():
                self.next_state = State.LOAD
            else:
                print("No saved data found, skipping")

        if key == KeyCodes.GROUP:
            self.group += 1
            self.old_points.append([])
            self.points.append([])
            self.old_triangles.append([])
            self.triangles.append([])
            self.transform_matrix.append([])
            self.old_circles.append([])
            self.old_lines.append([])
            self.circles.append([])
            self.lines.append([])

        if key == KeyCodes.DRAW:
            self.draw = not self.draw

        if key == KeyCodes.VISIBILITY:
            self.visibility = not self.visibility

        return True

    def save(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        # escape slashes in path
        name = self.video.split(".")[0]
        name = name.replace("/", ".")

        if not os.path.exists(f"output/{name}"):
            os.makedirs(f"output/{name}")

        np.save(
            f"output/{name}/group.npy",
            self.group,
        )
        for idx in range(self.group + 1):
            np.save(
                f"output/{name}/old_triangles_{idx}.npy",
                self.old_triangles[idx],
            )
            np.save(
                f"output/{name}/triangles_{idx}.npy",
                self.triangles[idx],
            )
            np.save(
                f"output/{name}/trans_matrix_{idx}.npy",
                self.transform_matrix[idx],
            )
        print("Saved!")

    def load(self):
        name = self.video.split(".")[0]
        name = name.replace("/", ".")

        if not os.path.exists(f"output/{name}"):
            return False

        self.group = np.load(f"output/{name}/group.npy")

        self.old_triangles = [[] for _ in range(self.group + 1)]
        self.triangles = [[] for _ in range(self.group + 1)]
        self.transform_matrix = [[] for _ in range(self.group + 1)]
        self.old_points = [[] for _ in range(self.group + 1)]
        self.points = [[] for _ in range(self.group + 1)]
        self.old_circles = [[] for _ in range(self.group + 1)]
        self.old_lines = [[] for _ in range(self.group + 1)]
        self.circles = [[] for _ in range(self.group + 1)]
        self.lines = [[] for _ in range(self.group + 1)]

        for idx in range(self.group + 1):
            self.old_triangles[idx] = np.load(f"output/{name}/old_triangles_{idx}.npy")
            self.triangles[idx] = np.load(f"output/{name}/triangles_{idx}.npy")
            # load np.array of lists
            self.transform_matrix[idx] = np.load(
                f"output/{name}/trans_matrix_{idx}.npy"
            )

        self.next_state = State.LOAD
        self.update_state()

        print("Loaded!")

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

    def _apply_warp(self, frame):
        bg = np.zeros_like(frame)
        bgs = []
        for idx in range(self.group + 1):
            if self.state != State.LOAD:
                self.transform_matrix[idx].clear()

            for iteration, (old_triangles, triangles) in enumerate(
                zip(
                    self.old_triangles[idx],
                    self.triangles[idx],
                )
            ):
                tri1 = np.float32(old_triangles)
                tri2 = np.float32(triangles)

                current_bg = np.zeros_like(frame)

                # Two rectangles are defined, each circumscribing the old and new triangles, respectively
                # Two rectangles are defined, each circumscribing the old and new triangles, respectively
                r1 = cv2.boundingRect(tri1)
                r2 = cv2.boundingRect(tri2)

                # The coordinates of both triangles are expressed relative to the coordinates of their respective
                # rectangles
                tri1Cropped = []
                tri2Cropped = []

                for i in range(0, 3):
                    tri1Cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
                    tri2Cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

                if self.state == State.LOAD:
                    warpMat = self.transform_matrix[idx][iteration]
                else:
                    # The getAffineTransform function of OpenCV [2 ] is employed to obtain the transformation matrix
                    # between the first and second triangle
                    warpMat = cv2.getAffineTransform(
                        np.float32(tri1Cropped), np.float32(tri2Cropped)
                    )
                    self.transform_matrix[idx].append(warpMat)

                # Crop input image
                img1Cropped = frame[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]

                # The transformation is then applied to the old rectangle to obtain the second one using the
                # warpAffine function.
                img2Cropped = cv2.warpAffine(
                    img1Cropped,
                    warpMat,
                    (r2[2], r2[3]),
                    None,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

                # A mask, sized according to the destination rectangle, is generated to retain only the pixels
                # related to the final triangle, setting the rest to zero
                mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
                cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

                img2Cropped = img2Cropped * mask

                # Copy triangular region of the rectangular patch to the output image
                current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = bg[
                    r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
                ] * ((1.0, 1.0, 1.0) - mask)

                # Subsequently, the new triangle is removed from the image and replaced by its warped version
                current_bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
                    bg[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Cropped
                )

                bgs.append(current_bg)
        bg = self.blend(frame, bgs)
        return bg

    def set_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame * self.precision)

    def render(self):
        cv2.namedWindow("Camera Dewarping")
        # cv2.namedWindow("selected area")

        self.cap = cv2.VideoCapture(self.video)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) // self.precision)
        cv2.createTrackbar(
            "frames",
            "Camera Dewarping",
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

            cv2.setMouseCallback("Camera Dewarping", self.on_mouse)
            self.get_triangles(frame)
            area = self.pre_warp_show(frame)

            if self.visibility:
                # create mask from area, where it is 0 set it to 255, else 0
                mask = np.zeros_like(area)
                mask[np.where(area == 0)] = 255
                mask[np.where(area != 0)] = 0
                # apply mask to frame
                frame = cv2.bitwise_and(frame, mask)
                frame = cv2.bitwise_or(frame, area)

            self.show(frame)

            self.update_state()
            key = cv2.waitKey(1) & 0xFF

        self.cap.release()
        cv2.destroyAllWindows()


@click.command()
@click.option(
    "--load",
    default=False,
    is_flag=True,
    help="Load presaved config (from output folder)",
)
@click.option("--scale", default=0.5, help="Window scale wrt. to video size")
@click.argument("file", type=click.Path(exists=True))
def main(file, load, scale):
    """
    Dewarping tool for cameras
    Press 'q' to exit
    Press 's' to save
    """
    a = Dewarping(file, load, scale)
    a.render()


if __name__ == "__main__":
    main()
