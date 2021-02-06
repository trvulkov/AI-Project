import numpy as np
import argparse
import cv2
import imutils
from skimage.filters import threshold_local

# all lists of points must maintain a consistent order: top-left, top-right, bottom-right, bottom-left
def order_points(pts): # takes a list of 4 points and returns them in the proper order
    rect = np.zeros((4, 2), dtype = "float32") # initialize a list that will contain the result

    sums = pts.sum(axis = 1) # sum x + y for each point
    rect[0] = pts[np.argmin(sums)] # the top-left point will have the smallest sum
    rect[2] = pts[np.argmax(sums)] # the bottom-right point will have the largest sum

    diffs = np.diff(pts, axis = 1) # subtract |x - y| for each point
    rect[1] = pts[np.argmin(diffs)] # the top-right point will have the smallest difference
    rect[3] = pts[np.argmax(diffs)] # the bottom-left will have the largest difference

    return rect

def four_point_transform(image, pts): # performs a perspective transformation to obtain a top-down view of an image
    rect = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    maxWidth = max(int(np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))),
                   int(np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))))
    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    maxHeight = max(int(np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))),
                    int(np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))))

    # construct the set of destination points to obtain a top-down view of the image
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped

def scan(image, debug, effect): # performs edge detection and returns the relevant part (the document) of the image 
    # we resize the image to have a height of 500 pixels in order to make the algorithm work better and faster
    ratio = image.shape[0] / 500.0 # compute the ratio of the old height to the new height
    orig = image.copy() # clone the image, so we can perform the scanning on the original image
    image = imutils.resize(image, height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Gaussian blurring to remove high frequency noise
    edged = cv2.Canny(blurred, 75, 200) # Canny edge detection

    if debug: # show the original image and an image with the results of the edge detection
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)

    # we assume that:
    # (1) the document to be scanned is the main focus of the image
    # (2) the document is rectangular, and thus will have four distinct edges
    # (3) thus, the largest contour in the image with exactly four points is our piece of paper to be scanned
    # so in order to find the part that we need, we find the contours in the edge-detected image, and pick the largest one with four edges
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # loop over the contours
    for contour in contours:
        # approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # if our approximated contour has four points, then we can assume that we have found our screen
        if len(approximated) == 4:
            screenContour = approximated
            break

    if debug: # show the contour (outline) of the piece of paper
        cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)

    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenContour.reshape(4, 2) * ratio)

    if effect:
        # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 33, offset = 10, method = "gaussian")
        warped = (warped > T).astype("uint8") * 255

    if debug: # show the scanned image
        cv2.imshow("Scanned", imutils.resize(warped, height = 500))
        cv2.waitKey()

    return warped

def parse(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required = True, help = "Path to the image")
    ap.add_argument("-debug", action="store_true", help = "Debug mode to display images of the various steps")
    ap.add_argument("-effect", action="store_true", help = "Convert the final image to grayscale and apply a threshold to give it the look of scanned paper")

    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    debug = args["debug"]
    effect = args["effect"]

    warped = scan(image, debug, effect)

    cv2.imwrite("output.png", warped)

parse()
