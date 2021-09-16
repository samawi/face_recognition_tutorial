import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# capture the video from the camera
face_image = face_recognition.load_image_file('images/samples/modi.jpg')

# find all facial landmarks in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(face_image)
print(face_landmarks_list)

# convert the numpy array image into pil image object
pil_image = Image.fromarray(face_image)

# convert the pil image to draw object
d = ImageDraw.Draw(pil_image, 'RGBA')

# loop 
for face_landmarks in face_landmarks_list:

    # make left and right eyebrows darker
    # polygon on top and line on bottom with dark color
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
    
    # add lipstick to top and bottom lips
    # using red polygons and lines filled with red
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 128), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128), width=8)

    # make left and right eyes red
    d.polygon(face_landmarks['left_eye'], fill=(255, 0, 0, 100))
    d.polygon(face_landmarks['right_eye'], fill=(255, 0, 0, 100))

    # eyeliner on left and right eyes as lines
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    
pil_image.show()