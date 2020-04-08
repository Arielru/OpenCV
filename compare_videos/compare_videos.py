import os, os.path
from datetime import datetime
from datetime import date
from skimage.measure import compare_ssim
import imutils
from cv2 import cv2 as cv2
import numpy as np
import time
import unittest

class Find_Differences(unittest.TestCase):

    __now = (datetime.now())
    __current_time = (str(__now.strftime(("%H_%M_%S"))))
    __today = (str(date.today()))
    __clips = []
    __folder = None
    __done_clip = None
    __counter = 1
    __nameA = None
    __nameB = None
    __directory = None
    __displa_time = 2000

    @classmethod
    def setUpClass(cls):
        Find_Differences.make_dir()
        Find_Differences.convert_to_jpg()

    @staticmethod
    def make_dir():
        for file in os.listdir(str('place_2_videos_to_compare')):
            if file.endswith(".mp4"):
                print("Files are at: {}".format(os.path.join(file)))
                os.mkdir(str(file))

            Find_Differences.__clips.append(file)

    @staticmethod
    def convert_to_jpg():
        counter = 1
        Find_Differences.__counter = 0
        for clip in Find_Differences.__clips:
            vidcap = cv2.VideoCapture('place_2_videos_to_compare\{}'.format(clip))
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite("{}\%d.jpg".format(clip) % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                print('Read frame # {}: '.format(Find_Differences.__counter), success)
                count += 1
                Find_Differences.__counter += 1

            counter += 1
            Find_Differences.__counter += 1

    def test_check_differences(self):
        counter_1 = 0
        print(print("Global counter: {}".format(self.__counter)))
        for x in range(int(self.__counter / 2) - 1):
            # load the two input images
            print("comparing images : {}".format(counter_1))
            imageA = cv2.imread("{}/{}.jpg".format(self.__clips[0], counter_1))
            self.__nameA = (counter_1)
            # cv2.imshow("Temp_A.jpg", imageA)
            imageB = cv2.imread("{}/{}.jpg".format(self.__clips[1], counter_1))
            self.__nameB = (counter_1)
            # cv2.imshow("Temp_B.jpg", imageB)
            # cv2.waitKey(0)
            # convert the images to grayscale
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = compare_ssim(grayA, grayB, full=True, multichannel=False)
            diff = (diff * 255).astype("uint8")
            print("SSIM: {}".format(score))

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            self.__counter += 1
            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)
                box1 = cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
                box2 = cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

            counter_1 += 1

            # show the output images
            images_1_2_h = np.hstack((imageA, imageB))
            cv2.namedWindow('images_1_2_h', cv2.WINDOW_NORMAL)
            cv2.imshow("images_1_2_h", images_1_2_h)

            if score != 1.0:
                date_and_time = (str(self.__today + '_' + self.__current_time))
                directory = ('delta_{}'.format(date_and_time))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                self.__directory = directory

                cv2.imwrite('{}/frame_{}.jpg'.format(directory, counter_1), images_1_2_h)


            # cv2.imshow("{}".format(self.__nameA), imageA)
            # cv2.imshow("{}".format(self.__nameB), imageB)
            # cv2.imshow("Diff", diff)
            # cv2.imshow("Thresh", thresh)
            cv2.waitKey(self.__displa_time)
            # time.sleep(5)
            cv2.destroyAllWindows()

        self.change_name()
        #folder = 'delta_2020-03-08_11_40_47'
        cwd = os.getcwd()
        print("current directory is: {}".format(cwd))
        print("Looking for: {}\{}".format(cwd, self.__directory))
        assert (os.path.exists("{}\{}".format(cwd, self.__directory))) == 0


    @staticmethod
    def change_name():
        for name in Find_Differences.__clips:
            os.rename(name, (name+'_'+ Find_Differences.__today + '_' + Find_Differences.__current_time))

if __name__ == '__main__':
    unittest.main()
