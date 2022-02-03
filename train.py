import os
import cv2 as cv
import numpy as np

# Names need to be the same as the folders in your training data
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Josh Gagnuss']

# Change this to your directory where your training data is stored 
DIR = r'C:\Users\joshg\OneDrive\Documents\GitHub\Facial Recognition with OPENCV\Faces\train'

# XML Document from OPENCV Github
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Initialise lists to store features & labels 
features =[]
labels = []

# Training function 
def create_train():
    for person in people: 
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('----------Model Is Trained----------')

# Create array from features & labels 
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Train Recognizer on features and labels
face_recognizer.train(features,labels) 

# Save data files required for recognition 
face_recognizer.save('trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)