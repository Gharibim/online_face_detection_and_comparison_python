from flask import Flask, render_template, Response
from mss import mss
import cv2
import sys
import numpy
import face_recognition



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    i = 1
    while i < 10:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + str(i) + b'\r\n')
        i += 1




def get_frame():
    camera_port = 0

    ramp_frames = 100

    # camera = cv2.VideoCapture(camera_port)  # this makes a web cam object
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    i = 1
    while True:

        retval, im = camera.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = im[y:y + h, x:x + w]
        # cv2.imshow('im', im)

        if (face_cascade.empty() == 'True'):
            print("Not empty")


        imgencode = cv2.imencode('.jpg', im)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
        i += 1

    del (camera)


@app.route('/calc')
def calc():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


def takesc():
    with mss() as sct:
        sct.shot()

@app.route('/takeScreenshot')
def takeScreenshot():
    takesc()
    return render_template('index.html')



def checkImg():
    try:
        known_image = face_recognition.load_image_file("monitor-1.png")
        unknown_image = face_recognition.load_image_file("me2.jpg")
        biden_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        if results[0] == 1:
            results = "Perfect Match between images"
        elif results[0] == 0:
            results = "The face detected does not match"
    except Exception:
        results = "No face was detected"

    return results

@app.route('/checkImage')
def checkImage():
    myResult = checkImg()
    return render_template('index.html', myResult= myResult)





if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)