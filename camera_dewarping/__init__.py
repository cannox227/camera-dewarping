import cv2


def show(frame):
    frame_downscaled = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow("preview", frame_downscaled)
    return cv2.waitKey(1) & 0xFF


VIDEOS = [
    "assets/left.mp4",
    "assets/right.mp4",
    "assets/center.mp4",
]


def switch_video(index):
    return cv2.VideoCapture(VIDEOS[index])


def main():
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(VIDEOS[0])
    # speed up video
    while True:
        ret, frame = cap.read()
        if not ret:
            # restart video if it ends
            cap = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        key_pressed = show(frame)
        
        if key_pressed == ord("q"):
            break
        elif key_pressed == ord("1"):
            cap = switch_video(0)
        elif key_pressed == ord("2"):
            cap = switch_video(1)
        elif key_pressed == ord("3"):
            cap = switch_video(2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
