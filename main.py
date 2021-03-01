from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3
import winsound

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
# cascade classifier

face_cascade = cv2.CascadeClassifier("haarcascade_frontal_default.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org

org = (50, 50)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[15], mouth[17])

    C = dist.euclidean(mouth[12], mouth[16])

    mar = (A + B) / (2.0 * C)

    return mar


def final_mar(shape):
    (Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    mmouth = shape[Start:End]

    mar = mouth_aspect_ratio(mmouth)

    return (mar, mmouth)


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

MAR_THRESH = 0.2
MAR_CONSEC_FRAMES = 15
COUNTER = 0

# print("-> Loading...")
# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# print("-> Turning on the camera....")
# vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
# time.sleep(1.0)

# fontScale
fontScale = 1

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

counter = 0
BEEP = False

# define threshold here
EAR_THRESHOLD = 0.25  # lessthan Threshold is close else open
# If eye is closed for 40 frame then we can say he is closing eye for long time
EYE_CLOSE_FRAME_THRESHOLD = 20


def EAR(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


def sound_alarm():
    # play an alarm sound
    if BEEP:
        winsound.Beep(440, 150)


def normalizeForVideoFunc(image):
    # image = cv2.imread("pic.jpeg")
    #  image = cv2.resize(image, (400, 300))
    s = image.shape
    # cv2.imshow("orginal", image)
    # print(image.shape)  # 600,800,3

    imageGrayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # imageGrayScale = cv2.convertScaleAbs(imageGrayScale, alpha=1, beta=-70)
    # cv2.imshow("gray", imageGrayScale)
    # print(imageGrayScale.shape)  # 600,800

    def showHist(image):
        histogramHolder = np.zeros(shape=(256, 1))
        s = image.shape
        for i in range(s[0]):
            for j in range(s[1]):
                intensity = image[i, j]
                histogramHolder[intensity, 0] = histogramHolder[intensity, 0] + 1
        return histogramHolder

    hist = showHist(imageGrayScale)
    # plt.plot(hist)
    # plt.show()

    hist = hist.reshape(1, 256)
    y = np.array([])
    y = np.append(y, hist[0, 0])

    for i in range(255):
        temp = hist[0, i + 1] + y[i]
        y = np.append(y, temp)

    y = np.round(y / ((s[0] * s[1])) * (255))
    # y = np.round((y/(s[0]*s[1]))*(255))

    for i in range(s[0]):
        for j in range(s[1]):
            intensity = imageGrayScale[i, j]
            imageGrayScale[i, j] = y[intensity]

    return imageGrayScale

    # hist = showHist(imageGrayScale)
    # plt.plot(hist)
    # plt.show()

    # cv2.imshow("after", imageGrayScale)
    # cv2.waitKey(0)


while True:
    # capture frame by frame
    ret, frame = cap.read()
    # print("fps is", cap.get(cv2.CAP_PROP_FPS))

    # if frame is read correctly ret is True
    if not ret:
        print("Cant read frame")
        break

    # gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=500)

    gray = normalizeForVideoFunc(frame)

    haarFaces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in haarFaces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if haarFaces is not ():
        cv2.putText(frame, "Face  detected",
                    (50, 80), font, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Face  notdetected",
                    (50, 80), font, 0.5, (0, 0, 255), 2)

    #################################################################

    faces = detector(gray)  # to detect face
    # print(faces)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # print(faces,"dlib faces")
        # if faces is not ():
        #     print("yeah")
        # else:
        #     print("error")

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # For highlight of eyes
        # landmark = predictor(gray, face)
        # for n in range(lStart, lEnd):
        #     x = landmark.part(n).x
        #     y = landmark.part(n).y
        #     cv2.circle(gray, (x, y), 2, (255, 0, 0), 2)

        # for n in range(42, 48):
        #     x = landmark.part(n).x
        #     y = landmark.part(n).y
        #     cv2.circle(gray, (x, y), 2, (255, 0, 0), 2)

        # for extraction of eye coordinates
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for
        # both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = EAR(leftEye)
        rightEar = EAR(rightEye)
        avgEar = (leftEar + rightEar) / 2

        if avgEar <= EAR_THRESHOLD:
            # print("closed")
            #  cv2.putText(frame, "Closed "+str(round(avgEar, 2))+"Counter= "+str(counter), org, font,
            #              fontScale, color, thickness, True)
            cv2.putText(frame, "Closed" + str(round(avgEar, 2)) + "Counter = " +
                        str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            counter = counter + 1
            if counter >= EYE_CLOSE_FRAME_THRESHOLD:
                BEEP = True
                sound_alarm()

        else:
            # print("Open")
            # cv2.flip(cv2.putText(frame, "Opened "+str(round(avgEar, 2))+"Counter= "+str(counter), org, font,
            #                      fontScale, color, thickness, True), 2)
            cv2.putText(frame, "Opened" + str(round(avgEar, 2)) + "Counter = " + str(counter), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            counter = 0
            BEEP = False
            sound_alarm()
            # mouth detection
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                mouthh = final_mar(shape)
                mar = mouthh[0]
                mmouth = mouthh[1]

                mouthHull = cv2.convexHull(mmouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                if mar > MAR_THRESH:
                    COUNTER += 1

                    if COUNTER >= MAR_CONSEC_FRAMES:
                        cv2.putText(frame, "YAWN ALERT!!!!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)  # Speed percent of voice-to-text (can go over 100)
                        engine.setProperty('volume', 1)  # Volume 0-1
                        engine.say('Be alerted sir.')
                        engine.runAndWait()

                else:
                    COUNTER = 0

                cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    # ASCII for ESC is 27 (to quit)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
