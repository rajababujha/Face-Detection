import cv2
import numpy as np

face_cascade = cv2.cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
##eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

id = input("Enter the ID: ")
cap = cv2.VideoCapture(0) #Here you can use only for the a single of using command
                          # img = cv2.imread('raja.jpg') and convert it to gray scale
                          # the OpenCV works good on gray scaled image
sampleNumber = 0          # To the specific number of image 
while True:
    
    _, img = cap.read()   # I used "_" but you can take any variable
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convetering into gray scale
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sampleNumber += 1
        cv2.imwrite("DataSet/User."+id+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)
            
                    
    cv2.imshow("Face",img)
    k = cv2.waitKey(1)
    if sampleNumber > 25:
        break

cap.release()
cv2.destroyAllWindows()

















##faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
##cam = cv2.VideoCapture(0)
##
##id = int(input("ID: "))
##while True:
##    ret,img = cam.read()
##    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##    faces = faceDetect.detectMultiScale(gray,1.3,5)
##    for (x,y,w,h) in faces:
##        cv2.imwrite("DataSet/User.",+str(id)+".jpg")
##        cv2.rectange(img,(x,y),(x+w,y+h),(0,0,255),2)
##        cv2.waitKey(100)
##    cv2.imshow("Face",img)
##    cv2.waitKey(1)
##
##cam.release()
##cv2.destroyAllWindows()
