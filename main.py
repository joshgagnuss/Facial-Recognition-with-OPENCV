import cv2 as cv 
import numpy as np

# Names of people in your training directory 
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Josh Gagnuss']

# Read in the XML document 
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Read in the trained model 
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')

# Read in the webcam stream 
video = cv.VideoCapture(0)

while True:
    try:
        ret,frame = video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x,y,w,h) in faces_rect:
            face = gray[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv.resize(face, (200,200))
            resized_face = np.expand_dims(resized_face, axis=0)

            label, confidence = face_recognizer.predict(gray)
        print(f'This is {people[label]} with a confidence of {confidence}')

        cv.putText(frame, str(people[label]), (40,40), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 1)
        cv.rectangle (frame, (x,y), (x+w, y+h), (0,255,0), 1)
        cv.imshow('Feed', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    except Exception as e:
        pass
    
video.release()
cv.destroyAllWindows()
            
    


