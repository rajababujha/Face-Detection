import cv2
import numpy as np
from PIL import Image

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('model.yml')

id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 0)

while True:
            
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        id ,conf = rec.predict(gray[y:y+h , x:x+w])
        if (id == 1):
            id = "Raja"
        elif (id == 2):
            id = "Papa"
        cv2.putText(img,str(id),(x,y+h),fontFace,fontScale,fontColor,2)
        cv2.imshow("Face",img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
         
