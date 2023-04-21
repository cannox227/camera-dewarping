from scipy.spatial import Delaunay
import numpy as np
import cv2


def show(frame):
    frame_downscaled = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("preview", frame_downscaled)
    return cv2.waitKey(1) & 0xFF


VIDEOS = [
    "assets/left.mp4",
    "assets/right.mp4",
    "assets/center.mp4",
]


def crop(frame, index):
    match index:
        case 0:
            return frame[:, 1400:2600]
        case 1:
            return frame[:, 1450:2550]
        case 2:
            return frame[:, 1400:2700]


def switch_video(index):
    return cv2.VideoCapture(VIDEOS[index])


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


def extract_some_points_from_lines(lines, n_points=50):
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append([x1, y1])
        points.append([x2, y2])
    points = np.array(points)
    # get random points
    random_indices = np.random.choice(points.shape[0], n_points, replace=False)
    random_points = points[random_indices, :]
    return random_points


def main():
    cv2.namedWindow("preview")
    cap = switch_video(0)
    current_source = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # restart video if it ends
            cap = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame = crop(frame, current_source)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = auto_canny_edge_detection(gaussian)
        minLineLength = 50
        maxLineGap = 10
        lines = cv2.HoughLinesP(
            edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap
        )
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # get some points from the lines
        points = extract_some_points_from_lines(lines)

        # triangulate the points
        tri = Delaunay(points)
        
        # draw the triangulation
        for triangle in tri.simplices:
            pt1 = tuple(points[triangle[0]])
            pt2 = tuple(points[triangle[1]])
            pt3 = tuple(points[triangle[2]])
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.line(frame, pt2, pt3, (0, 0, 255), 2)
            cv2.line(frame, pt3, pt1, (0, 0, 255), 2)

        key_pressed = show(frame)

        if key_pressed == ord("q"):
            break
        elif key_pressed == ord("1"):
            cap = switch_video(0)
            current_source = 0
        elif key_pressed == ord("2"):
            cap = switch_video(1)
            current_source = 1
        elif key_pressed == ord("3"):
            cap = switch_video(2)
            current_source = 2
        elif key_pressed == ord(" "):
            # pause or unpause
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
