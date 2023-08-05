import numpy as np                                                                                     
import cv2        

def overlap(image_1, image_2):
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    # if pixel in pixel_1 and pixel_2 have a 10% difference, set output pixel to (pixel_1+pixel_2)/2
    image = np.zeros_like(image_1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # if image_1[i][j] < 150 and image_2[i][j] < 150:
            #     continue
            if abs(image_1[i][j] - image_2[i][j]) < 2:
                image[i][j] = image_1[i][j]
            else:
                image[i][j] = (image_1[i][j] + image_2[i][j]) / 2
    return image

def main():
    image_1 = cv2.imread("output/0.jpg")
    image_2 = cv2.imread("output/1.jpg")
    image_1 = cv2.GaussianBlur(image_1, (5, 5), 0)
    image_2 = cv2.GaussianBlur(image_2, (5, 5), 0)
    # shift image_1 to the right
    image_1 = np.roll(image_1, 40, axis=1)
    image_2 = np.roll(image_2, -40, axis=1)
    image = overlap(image_1, image_2)
    # image = cv2.bitwise_or(image_1, image_2)

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("preview", image)
    cv2.waitKey(0)