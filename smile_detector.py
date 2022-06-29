from tabnanny import verbose
from build_model import BuildModel
import keras
import cv2
import numpy as np

TRESHOLD = 0.8


def main(debug=False):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    my_model = BuildModel()
    my_model.load_model('binary_model2')


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
           
            prediction = my_model.single_prediction(face)
            
            rec_color = (255, 0, 0)
            if prediction > TRESHOLD:
                rec_color = (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+  face_width, y + face_height), rec_color, 2)
            if debug:
                cv2.rectangle(frame, (0,0), (100, 40), (255,255,255), -1)
                frame = cv2.putText(
                    img = frame,
                    text = str(np.round(prediction[0][0], 3)),
                    org = (10, 30),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1.5,
                    color = (0, 0, 0),
                    thickness = 3
                )

        cv2.imshow('Smile!', frame)

        key = cv2.waitKey(50)
        # esc button
        if key == 27:
            break

    capture.release()


if __name__ == '__main__':
    main(debug=True)
    