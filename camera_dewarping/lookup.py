import cv2
import sys
import numpy as np

def lookup(image_1, image_1_new, image_2, image_2_new):
    # compute intersection
    image_intersection = cv2.bitwise_and(image_1_new, image_2_new)

    # make only intersection black, and rest white
    image_mask = np.zeros_like(image_intersection)
    image_mask[np.where((image_intersection == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    image_mask_inv = cv2.bitwise_not(image_mask)
    
    image_1_new = cv2.bitwise_and(image_1_new, image_mask)
    image_2_new = cv2.bitwise_and(image_2_new, image_mask)

    image = cv2.add(image_1_new, image_2_new)

    # iterate over white pixel in image_mask_inv
    pixels = np.where((image_mask_inv == [255, 255, 255]).all(axis=2))
    print(pixels)

    return image

def main():
    image_1 = cv2.imread("output/0.jpg")
    image_2 = cv2.imread("output/1.jpg")
    image_1_new = np.roll(image_1, 40, axis=1)
    image_2_new = np.roll(image_2, -40, axis=1)

    image = lookup(image_1, image_1_new, image_2, image_2_new)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("preview", image)
    cv2.waitKey(0)