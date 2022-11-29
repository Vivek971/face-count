import cv2

# Load the cascade
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capture from cam, used as video input
cap = cv2.VideoCapture(0)


while True:
    # read each frame
    _,frame = cap.read()

    # now convert image into grayscales image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect the faces
    faces = face_classifier.detectMultiScale(gray,1.1,4)

    # draw rect around faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    # display
    cv2.imshow("Frame",frame)

    k=cv2.waitKey(30) & 0xff
    if k== 27:
        break
print(faces)
print("found {0} faces".format(len(faces)))
cap.release()
# cv2.destroyAllWindows()
