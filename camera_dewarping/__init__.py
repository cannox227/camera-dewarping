from .camera import main
main()
# import numpy as np
# import cv2
# import time


# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         global POINTS
#         POINTS[CURRENT_SOURCE].append((int(x // SCALE), int(y // SCALE)))


# def switch_video(index):
#     return cv2.VideoCapture(VIDEOS[index])


# def triangulate(frame, transform=False):
#     overlay = np.zeros_like(frame)
#     global TRANSFORMS
#     TRANSFORMS[CURRENT_SOURCE] = []
#     if len(POINTS[CURRENT_SOURCE]) > 3:

#         tri = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
#         # draw triangles
#         for point in POINTS[CURRENT_SOURCE]:
#             tri.insert(point)

#         # convert triangles to int
#         triangles = tri.getTriangleList().astype(np.int32)

#         # draw delaunay triangles
#         for triangle in triangles:
#             pt1 = tuple(triangle[0:2])
#             pt2 = tuple(triangle[2:4])
#             pt3 = tuple(triangle[4:6])

#             mask = np.zeros_like(frame)
#             # draw triangle and fill it
#             cv2.drawContours(mask, [np.array([pt1, pt2, pt3])], 0, (255, 255, 255), -1)

#             if not transform:
#                 MASKS[CURRENT_SOURCE].append(mask)
#             else:
#                 edit = np.bitwise_and(frame, mask)
#                 edit = bilinear_interpolation(edit, pt1, pt2, pt3)
#                 TRANSFORMS[CURRENT_SOURCE].append(edit)

#             cv2.line(overlay, pt1, pt2, (0, 0, 255), 5)
#             cv2.line(overlay, pt2, pt3, (0, 0, 255), 5)
#             cv2.line(overlay, pt3, pt1, (0, 0, 255), 5)

#     return overlay


# def merge_masks(frame, transform=False):
#     full = np.zeros_like(frame)
#     if transform:
#         for mask in TRANSFORMS[CURRENT_SOURCE]:
#             full = cv2.bitwise_or(full, mask)
#         return full
#     else:
#         for mask in MASKS[CURRENT_SOURCE]:
#             full = cv2.bitwise_or(full, mask)
#         return full

# def get_n_mask(frame, index=0):
#     # get random mask from MASKS
#     if len(MASKS[CURRENT_SOURCE]) == 0:
#         return np.zeros_like(frame)
#     return MASKS[CURRENT_SOURCE][index]

# def merge_horizontal(frame1, frame2):
#     return np.hstack((frame1, frame2))

# positive = False

# def bilinear_interpolation(frame, pt1, pt2, pt3):
#     pt4 = (pt1[0]+np.random.randint(50, 150), pt1[1])
#     pt5 = (pt2[0]+np.random.randint(50, 150), pt2[1])
#     pt6 = (pt3[0]+np.random.randint(50, 150), pt3[1])
#     src = np.array([pt1, pt2, pt3], dtype=np.float32)
#     dst = np.array([pt4, pt5, pt6], dtype=np.float32)
#     M = cv2.getAffineTransform(src, dst)
#     return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    


# def main():
#     cv2.namedWindow("preview")
#     cap = switch_video(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             # restart video if it ends
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         frame = crop(frame)

#         overlay = triangulate(frame)

#         merge = merge_masks(frame)

#         cv2.setMouseCallback("preview", mouse_callback)

#         # add overlay and merge to frame
#         # overlay = cv2.bitwise_or(overlay, merge)
#         # frame = cv2.bitwise_and(frame, overlay)
#         mask = cv2.bitwise_and(frame, merge)

#         combined = frame

#         triangulate(combined, transform=True)

#         merge_transform = merge_masks(combined, transform=True)
#         # mask_transform = cv2.bitwise_and(combined, merge_transform)

#         combined = merge_horizontal(combined, merge_transform)

#         # combined = merge_horizontal(combined, trans)


#         key_pressed = show(combined)
#         global CURRENT_SOURCE

#         # time.sleep(1)

#         if key_pressed == ord("q"):
#             break
#         elif key_pressed == ord("1"):
#             cap = switch_video(0)
#             CURRENT_SOURCE = 0
#         elif key_pressed == ord("2"):
#             cap = switch_video(1)
#             CURRENT_SOURCE = 1
#         elif key_pressed == ord("3"):
#             cap = switch_video(2)
#             CURRENT_SOURCE = 2
#         elif key_pressed == ord(" "):
#             # pause or unpause
#             cv2.waitKey(0)
#         # check for backspace
#         elif key_pressed == 8:
#             # remove all points
#             POINTS[CURRENT_SOURCE] = []
#         # check for enter
#         elif key_pressed == ord("\r"):
#             break
#         elif key_pressed == ord("s"):
#             for index, im in enumerate(TRANSFORMS[CURRENT_SOURCE]):
#                 cv2.imwrite(f"output/{CURRENT_SOURCE}_{index}.png", im)

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
