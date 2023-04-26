from scipy.spatial import Delaunay
import numpy as np
import cv2

CURRENT_SOURCE = 0

SCALE = 0.5

VIDEOS = [
    "assets/left.mp4",
    "assets/right.mp4",
    "assets/center.mp4",
]

POINTS = {0: [], 1: [], 2: []}


def show(frame):
    frame_downscaled = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    cv2.imshow("preview", frame_downscaled)
    return cv2.waitKey(1) & 0xFF


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global POINTS
        POINTS[CURRENT_SOURCE].append((int(x // SCALE), int(y // SCALE)))


def crop(frame):
    match CURRENT_SOURCE:
        case 0:
            return frame[:, 1400:2600]
        case 1:
            return frame[:, 1450:2550]
        case 2:
            return frame[:, 1400:2700]


def switch_video(index):
    return cv2.VideoCapture(VIDEOS[index])


def triangulate(frame):
    if len(POINTS[CURRENT_SOURCE]) > 3:
        tri = Delaunay(POINTS[CURRENT_SOURCE])
        for triangle in tri.simplices:
            pt1 = tuple(POINTS[CURRENT_SOURCE][triangle[0]])
            pt2 = tuple(POINTS[CURRENT_SOURCE][triangle[1]])
            pt3 = tuple(POINTS[CURRENT_SOURCE][triangle[2]])
            cv2.line(frame, pt1, pt2, (0, 0, 255), 5)
            cv2.line(frame, pt2, pt3, (0, 0, 255), 5)
            cv2.line(frame, pt3, pt1, (0, 0, 255), 5)

    return frame


def main():
    cv2.namedWindow("preview")
    cap = switch_video(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            # restart video if it ends
            cap = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame = crop(frame)

        frame = triangulate(frame)

        cv2.setMouseCallback("preview", mouse_callback)

        key_pressed = show(frame)

        global CURRENT_SOURCE

        if key_pressed == ord("q"):
            break
        elif key_pressed == ord("1"):
            cap = switch_video(0)
            CURRENT_SOURCE = 0
        elif key_pressed == ord("2"):
            cap = switch_video(1)
            CURRENT_SOURCE = 1
        elif key_pressed == ord("3"):
            cap = switch_video(2)
            CURRENT_SOURCE = 2
        elif key_pressed == ord(" "):
            # pause or unpause
            cv2.waitKey(0)
        # check for enter
        # elif key_pressed == ord("\r"):

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
