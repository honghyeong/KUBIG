# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import imutils
import argparse

# construct the argument parse and parse the arguments
ap=argparse.ArgumentParser(description='Use Model to detect smile')
ap.add_argument('-m','--model',required=True,help='path to pre-trained smile detector')
ap.add_argument('-v','--video',required=False,help='path to the optional video file')
args=vars(ap.parse_args())


# load models
model=load_model(args['model'])
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# perform initialization for webcam or video
# if a video path was not supplied, grab the reference to the webcam
if not args.get('video',False):
    camera=cv2.VideoCapture(0)
# otherwise, load the video
else:
    camera=cv2.VideoCapture(args['video'])

# keep looping
while True:
    # grab the current frame
    (grabbed,frame)=camera.read()

    # if using video and did not grab a frame, then end of video, break
    if args.get('video') and not grabbed:
        break

    # resize and clone original frame to draw on it later
    frame=imutils.resize(frame,width=800)
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) : # data decline
    frameClone=frame.copy()

    rects=detector.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(64,64),
                                    flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the ace bounding boxes

    for (fX,fY,fW,fH) in rects:
        # extract the ROT of the face of image
        roi=frame[fY:fY+fH,fX:fX+fW]
        # resize 64^64
        roi=cv2.resize(roi,(64,64))
        roi=roi.astype('float')/255
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)

        # determine the probs of smiling and set label
        smiling = model.predict(roi)[0]
        label = "Smiling" if smiling > 0.5 else "Not Smiling"

        # display the label and and bounding box rectangle on the output frame
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # show detect face along with smiling not smiling labels
    cv2.imshow("Face", frameClone)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cleanup camera and close any open windows
camera.release()
cv2.destroyAllWindows()