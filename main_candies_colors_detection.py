import os

import cv2
import numpy as np


class image_etudiee :
    def __init__(self):
        self.nom, self.nb_couleurs, self.list_couleurs = "", 0, list()
    def __init__(self, nom):
        self.nom = nom
        self.nb_couleurs, self.list_couleurs = 0, list ( )
    def __init__(self, nom, nb_couleurs):
        self.nom = nom
        self.nb_couleurs = nb_couleurs
        self.list_couleurs = list()
    def __init__(self, nom, nb_couleurs, list_couleurs):
        self.nom = nom
        self.nb_couleurs = nb_couleurs
        self.list_couleurs = list_couleurs

    def disp(self):
        cv2.imshow("images/"+self.nom)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read(self):
        return cv2.imread("images/"+self.nom)


    def detect_colors(self):
        im_list = list()
        for i in range(self.nb_couleurs):
            color = self.list_couleurs[i][0]
            lower_color = self.list_couleurs[i][1]
            upper_color = self.list_couleurs [i][2]
            im_list.append(self.detect_one_color(lower_color, upper_color))

    def detect_one_color(self, lower_color, upper_color):
        frame = self.read()
        mask = cv2.inRange (hsv, lower_color, upper_color)
        return cv2.bitwise_and (frame, frame, mask=mask)


if __name__ == '__main__':
    #cap = cv2.VideoCapture (0)
    image = image_etudiee("mms_peanut.png", 6, ['red','green', 'brown','orange', 'blue', 'yellow' ])

    xx = cv2.imread("images/mms_peanut.png")
    print(np.shape(xx))
    # cv2.imshow("images/mms_peanut.png", xx)

    frame = xx
    hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)

    # For blue candies
    lower_blue = np.array ([ 100, 40, 40 ])
    upper_blue = np.array ([ 140, 255, 255 ])
    mask = cv2.inRange (hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and (frame, frame, mask=mask)

    res = cv2.resize (res, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('res', res)
    mask = cv2.resize (mask, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask', mask)

    # For red candies
    lower_red = np.array ([ 0, 50, 40 ])
    upper_red = np.array ([ 5, 250, 250 ])
    mask_red = cv2.inRange (hsv, lower_red, upper_red)
    res_red = cv2.bitwise_and (frame, frame, mask=mask_red)

    res_red = cv2.resize (res_red, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('res_red', res_red)
    mask_red = cv2.resize (mask_red, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask_red', mask_red)

    # For yellow candies

    lower_yellow = np.array ([20, 180, 150])
    upper_yellow = np.array ([50, 250, 250])
    mask_yellow = cv2.inRange (hsv, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and (frame, frame, mask=mask_yellow)

    res_yellow = cv2.resize (res_yellow, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('res_yellow', res_yellow)
    mask_yellow = cv2.resize (mask_yellow, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask_yellow', mask_yellow)

    # For orange candies

    lower_orange = np.array ([ 10, 190, 200])
    upper_orange = np.array ([ 20, 255, 255])
    mask_orange = cv2.inRange (hsv, lower_orange, upper_orange)
    res_orange = cv2.bitwise_and (frame, frame, mask=mask_orange)

    res_orange = cv2.resize (res_orange, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('res_orange', res_orange)
    mask_orange = cv2.resize (mask_orange, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask_orange', mask_orange)


    # For green candies

    lower_green = np.array([ 45, 100, 100])
    upper_green = np.array([ 75, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(frame, frame, mask = mask_green)

    res_green = cv2.resize(res_green, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('res_green', res_green)
    mask_green = cv2.resize (mask_green, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask_green', mask_green)

    # For brown candies

    lower_brown = np.array ([ 20, 100, 100 ])
    upper_brown = np.array ([ 30, 200, 200 ])
    mask_brown = cv2.inRange (hsv, lower_brown, upper_brown)
    res_brown = cv2.bitwise_and (frame, frame, mask=mask_brown)

    res_brown = cv2.resize (res_brown, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('res_brown', res_brown)
    mask_brown = cv2.resize (mask_brown, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow ('mask_brown', mask_brown)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

