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

MASKS = {0: [], 1: [], 2: []}

LAST_POINTS = {0: 0, 1: 0, 2: 0}


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
    overlay = np.zeros_like(frame)
    if len(POINTS[CURRENT_SOURCE]) > 3:
        tri = Delaunay(POINTS[CURRENT_SOURCE])
        for triangle in tri.simplices:
            pt1 = tuple(POINTS[CURRENT_SOURCE][triangle[0]])
            pt2 = tuple(POINTS[CURRENT_SOURCE][triangle[1]])
            pt3 = tuple(POINTS[CURRENT_SOURCE][triangle[2]])
                
            mask = np.zeros_like(frame)
            # draw triangle and fill it
            cv2.drawContours(mask, [np.array([pt1, pt2, pt3])], 0, (255, 255, 255), -1)

            # crop = mask[
            #     min(pt1[1], pt2[1], pt3[1]) : max(pt1[1], pt2[1], pt3[1]),
            #     min(pt1[0], pt2[0], pt3[0]) : max(pt1[0], pt2[0], pt3[0]),
            # ]
            MASKS[CURRENT_SOURCE].append(mask)

            cv2.line(overlay, pt1, pt2, (0, 0, 255), 5)
            cv2.line(overlay, pt2, pt3, (0, 0, 255), 5)
            cv2.line(overlay, pt3, pt1, (0, 0, 255), 5)    

    return overlay

def merge_masks(frame):
    full = np.zeros_like(frame)
    for mask in MASKS[CURRENT_SOURCE]:
        full = cv2.bitwise_or(full, mask)
    return cv2.bitwise_not(full)


def main():
    cv2.namedWindow("preview")
    cap = switch_video(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            # restart video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = crop(frame)

        overlay = triangulate(frame)

        merge = merge_masks(frame)

        cv2.setMouseCallback("preview", mouse_callback)
        
        # add overlay and merge to frame
        overlay = cv2.bitwise_or(overlay, merge)
        frame = cv2.bitwise_and(frame, overlay)


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
        # check for backspace
        elif key_pressed == 8:
            # remove all points
            POINTS[CURRENT_SOURCE] = []
        # check for enter
        elif key_pressed == ord("\r"):
            break
    

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
