import cv2 as cv  # pylint: disable=no-member
import numpy as np
# pylint: disable=no-member

width = 640
height = 480
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


# def function to find contours


def findcontours(img, imgContour):
    contours, hierarchy = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
# RETER EXTERNAL USE WHEN WE ARE FINDING OUTER CONTOURS

    # find area
    for cnt in contours:
        area = cv.contourArea(cnt)
        areamin = cv.getTrackbarPos("area", "parameters")
        if area > areamin:
            cv.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri = cv.arcLength(cnt, True)
            # print(peri)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            cv.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv.putText(imgContour, f"Points: {len(approx)}", (x + w + 20, y + 20),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(imgContour, f"Area: {int(area)}", (x + w + 20, y + 45),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # cv.putText(imgContour, objectType,
            # (x + (w // 2) - 10, y + (h // 2) - 10),
            # cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)


def empty(x):
    pass


cv.namedWindow('parameters')
cv.resizeWindow("parameters", 640, 480)
cv.createTrackbar("threshold1", "parameters", 200, 255, empty)
cv.createTrackbar("threshold2", "parameters", 200, 255, empty)
cv.createTrackbar("area", "parameters", 500, 10000, empty)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(
                        imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success, img = cap.read()  # 1st read
    img = cv.resize(img, (400, 400))

    imgContour = img.copy()
    # img = img.resize(100, 100)
    imgBlur = cv.GaussianBlur(img, (7, 7), 1)  # 2nd blur the image
    # 3rd blur img to gray image
    imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

    # 4th add canny image to find outline

    # it will show trachbar of threshold to adjust
    threshold1 = cv.getTrackbarPos("threshold1", "parameters")
    threshold2 = cv.getTrackbarPos("threshold2", "parameters")
    imagcanny = cv.Canny(imgGray, threshold1, threshold2)

    # 5th to remove noise use dilate
    # Apply dilation to the Canny image
    imgDilate = cv.dilate(imagcanny, (7, 7), iterations=1)
    findcontours(imgDilate, imgContour)  # 6th call contour function

    stackimages = stackImages(0.8, ([img, imgGray, imagcanny], [
                              imgDilate, imgContour, imgContour]))
    cv.imshow("stack", stackimages)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
# pylint: enable=no-member   (this comment is added to remove any pylint red lines error)
