# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import warnings
from os import system, name
import pyfiglet 

warnings.simplefilter(action='ignore', category=FutureWarning)

# Cleaning Shell
if name == 'nt':
        _ = system('cls')
else:
        _ = system('clear') 

print(pyfiglet.figlet_format("Blink & Yawn Detection", font = "digital" ))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# define two variables for yawn count and status
yawns = 0
yawn_status = False

# initialize the frame counters and the blinks number of blinks
COUNTER = 0
blinks = 0

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] Starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True

def eye_aspect_ratio(eye):
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

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    # im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(im, pos, 3, color=(0, 255, 0))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, -1
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance
 
# loop over frames from the video stream
print("[INFO] Initiating Detection Process...")
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	if np.shape(frame) != ():
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale frame
                rects = detector(gray, 0)

                # loop over the face detections
                for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        
                        if leftEye.all() and rightEye.all():
                                leftEAR = eye_aspect_ratio(leftEye)
                                rightEAR = eye_aspect_ratio(rightEye)

                                # average the eye aspect ratio together for both eyes
                                ear = (leftEAR + rightEAR) / 2.0

                                # compute the convex hull for the left and right eye, then
                                # visualize each of the eyes
                                leftEyeHull = cv2.convexHull(leftEye)
                                rightEyeHull = cv2.convexHull(rightEye)
                                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                                # check to see if the eye aspect ratio is below the blink
                                # threshold, and if so, increment the blink frame counter
                                if ear < EYE_AR_THRESH:
                                        COUNTER += 1

                                # otherwise, the eye aspect ratio is not below the blink
                                # threshold
                                else:
                                        # if the eyes were closed for a sufficient number of
                                        # then increment the blinks number of blinks
                                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                                blinks += 1

                                        # reset the eye frame counter
                                        COUNTER = 0

                                # draw the blinks number of blinks on the frame along with
                                # the computed eye aspect ratio for the frame
                                cv2.putText(frame, "Blink Count: {}".format(blinks), (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, "Yawn Count: {}".format(yawns), (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                image_landmarks, lip_distance = mouth_open(frame)
                                prev_yawn_status = yawn_status
                                prev_lip_distance = lip_distance
                                if lip_distance > 25 or lip_distance == -1:
                                        yawn_status = True
                                else:
                                        yawn_status = False
                                if prev_yawn_status == True and yawn_status == False:
                                        yawns += 1
                                
                                prev_lip_distance = lip_distance
         
                # show the frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
         
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break
print("[INFO] Detection Process Completed...")
print("[INFO] Closing video stream thread...\n")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print("[RESULT]\n")
print("Yawn Count = " + str(yawns))
print("Blink Count = " + str(blinks) + "\n\n")
