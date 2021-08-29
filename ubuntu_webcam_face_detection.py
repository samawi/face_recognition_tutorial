import face_recognition
import cv2
#from imutils.video import WebcamVideoStream
#import imutils

webcam_video_stream = cv2.VideoCapture(0)
# vs = WebcamVideoStream(src=0).start()

# set resolution (3 and 4)
webcam_video_stream.set(3, 320)
webcam_video_stream.set(4, 240)

# set framerate (5)
# webcam_video_stream.set(5, 15)

all_face_locations = []

while True:
    # current_frame = vs.read()
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    # all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='cnn')

    # for index, current_face_location in enumerate(all_face_locations):
    #     top_pos, right_pos, bottom_pos, left_pos = current_face_location
    #     top_pos *= 4
    #     right_pos *= 4
    #     bottom_pos *= 4
    #     left_pos *= 4

    #     cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    cv2.imshow('Webcam Video', current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
# vs.stop()