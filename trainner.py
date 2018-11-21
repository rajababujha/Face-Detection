import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "DataSet" # Creating a folder named of DataSet in the same directory

def getImagesWithID(path):
    for f in os.listdir(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg , dtype=np.uint8)# Converting the image to np array
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) #Taking the id which was provided at the capturing the data from face
            faces.append(faceNp)
            print(ID)
            IDs.append(ID)
            cv2.imshow("Training",faceNp)
            cv2.waitKey(5)
        return IDs,faces

Ids, faces = getImagesWithID(path)
recognizer.train(faces,np.array(Ids)) # Training the model on the taking dataset
recognizer.write('model.yml')
cv2.destroyAllWindows()
