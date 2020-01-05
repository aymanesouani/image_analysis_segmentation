import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing

if __name__ == "__main__" :
    # Read image
    im_read = cv2.imread("images/mms_peanut.png")

    # Change spacecolor from RGB to LAB
    lab_im = cv2.cvtColor(im_read, cv2.COLOR_BGR2Lab)
    print(lab_im.shape)
    # Choose bands to work with
    ab = lab_im
    print(np.shape(ab[:,:,0]))
    ab[:,:,0] = 255 * np.ones(np.shape(ab[:,:,0]))

    # Set the number of colors (number of clusters)
    nb_colors = 2

    # define criteria
    eps = 0.02
    max_iter = 500
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    print(criteria)
    print(ab.shape)
    # Apply kmean algorithm on the image for classification
    # Normalize image before applying k-means
    '''for i in range(3) :
        ab[:][:][i] = preprocessing.normalize(ab[:][:][i], norm='max')
'''
    pixel_values = ab.reshape((-1, 3))
    pixel_values = np.float32 (pixel_values)
    print(pixel_values.shape)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels1, centers1 = cv2.kmeans (pixel_values, nb_colors, None, criteria, 10, flags)

    centers = np.uint8(centers1)
    segmented_image1 = centers[labels1.flatten()]
    segmented_image1 = segmented_image1.reshape(ab.shape)
    # show the image
    plt.figure(1)
    grayImage1 = cv2.cvtColor (segmented_image1, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayImage1)
    plt.show()

    # Change spacecolor from RGB to LAB
    hsv_im = cv2.cvtColor (im_read, cv2.COLOR_BGR2HSV)
    print (hsv_im.shape)
    # Choose bands to work with
    # We work here with all the bands
    # hsv_im[:][:][1] = 255 * np.ones(np.shape(hsv_im[:][:][2]))
    # Set the number of colors (number of clusters)
    # Same number of colors
    nb_colors = 3



    # define criteria
    eps = 0.0002
    max_iter = 3000
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    # Normalize image
    '''for i in range(3) :
        hsv_im[:][:][i] = preprocessing.normalize(hsv_im[:][:][i], norm='max')
'''
    # Apply kmean algorithm on the image for classification
    pixel_values = hsv_im.reshape ((-1, 3))
    pixel_values = np.float32 (pixel_values)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels2, centers2 = cv2.kmeans (pixel_values, nb_colors, None, criteria, 10, flags)

    centers = np.uint8 (centers2)
    segmented_image2 = centers [ labels2.flatten ( ) ]
    segmented_image2 = segmented_image2.reshape (hsv_im.shape)
    # show the image
    plt.figure (2)
    grayImage2 = cv2.cvtColor (segmented_image2, cv2.COLOR_BGR2GRAY)
    plt.imshow (grayImage2)
    plt.show ()

    # show the sum of images
    plt.figure (3)
    plt.imshow (grayImage2 + grayImage1)
    plt.show ( )

    plt.figure(4)
    grayImage3 = cv2.cvtColor (segmented_image2 + segmented_image1, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayImage3)
    plt.show()



