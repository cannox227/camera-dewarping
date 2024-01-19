import numpy as np
import cv2


def overlap(image_1, image_2):
    image_intersection = cv2.bitwise_and(image_1, image_2)
    image_mask = np.zeros_like(image_intersection)
    image_mask[np.where(image_intersection >= 1)] = 255
    image_mask_inv = np.ones_like(image_mask) * 255
    image_mask_inv[np.where(image_mask >= 1)] = 0

    image_overlap = np.zeros_like(image_1)
    image_overlap = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)

    image_common = cv2.bitwise_and(image_overlap, image_mask)

    image_final = image_common.copy()

    image_1_new = cv2.bitwise_and(image_1, image_mask_inv)
    image_2_new = cv2.bitwise_and(image_2, image_mask_inv)

    image_final = cv2.addWeighted(image_final, 1, image_1_new, 1, 0)
    image_final = cv2.addWeighted(image_final, 1, image_2_new, 1, 0)

    return image_final


def main():
    image_1 = cv2.imread("output/0.jpg")
    image_2 = cv2.imread("output/1.jpg")
    image_1 = np.roll(image_1, 40, axis=1)
    image_2 = np.roll(image_2, -40, axis=1)

    image = overlap(image_1, image_2)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("preview", image)
    cv2.waitKey(0)
