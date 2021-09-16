import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np
# capture the video from the camera
webcam_video_stream = cv2.VideoCapture(0)
webcam_video_stream.set(3,640)
webcam_video_stream.set(4,320)
all_face_locations = []
while True:
    ret, current_frame = webcam_video_stream.read()
    # current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)

    # find all facial landmarks in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(current_frame)

    # convert the numpy array image into pil image object
    pil_image = Image.fromarray(current_frame)

    # convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)

    # loop for every face
    index = 0

    while index < len(face_landmarks_list):

        # loop 
        for face_landmarks in face_landmarks_list:

            # join each face landmark points
            d.line(face_landmarks['chin'], fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eyebrow'], fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eyebrow'], fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_bridge'], fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_tip'], fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eye'], fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eye'], fill=(255,255,255), width=2)
            d.line(face_landmarks['top_lip'], fill=(255,255,255), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(255,255,255), width=2)

        index += 1

    # convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert('RGB')
    rgb_opencv_image = np.array(pil_image)

    # convert RGB to BGR
    bgr_opencv_image = cv2.cvtColor(rgb_opencv_image, cv2.COLOR_RGB2BGR)
    bgr_opencv_image = bgr_opencv_image[:, :, ::-1].copy()

    # showing the current face
    cv2.imshow("Webcam video", bgr_opencv_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()