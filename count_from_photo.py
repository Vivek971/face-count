import cv2

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread('img2.jpg')

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(imgGray,1.1,4)
i =0
for (x,y,w,h) in faces:
    i=i+1
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)
    cv2.putText(img, 'F-' + str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('img2.jpg',img)
print('Number of faces are {0}'.format(len(faces)))
cv2.waitKey(0)