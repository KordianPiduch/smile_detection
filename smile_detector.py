from tabnanny import verbose
import keras
import cv2
import numpy as np

TRESHOLD = 0.9

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
model = keras.models.load_model('./saved_models/binary_model2')


while True:
    _, frame = capture.read()
    # grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50,50)
    )

    for x, y, face_width, face_height in faces:
        start_point = (x, y)
        end_point = (x + face_width, y + face_height)

        crop_frame = frame[y:y+face_height, x:x+face_width]

        face = cv2.resize(crop_frame, (178,178))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255

        prediction = model.predict(np.array([face, ]), verbose=True)
        
        rec_color = (255, 0, 0)
        if prediction > TRESHOLD:
            rec_color = (0, 255, 0)
        

        cv2.rectangle(frame, (x, y), (x+  face_width, y + face_height), rec_color, 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(50)
    # esc button
    if key == 27:
        break

capture.release()