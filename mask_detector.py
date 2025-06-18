import cv2 # type: ignore
import numpy as np

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

maskColors=[
    ((90, 50, 70), (128, 255, 255)), 
    ((36, 50, 70), (89, 255, 255)),  
    ((0, 0, 180), (180, 50, 255)),
    ((0, 0, 0), (180, 255, 50))
]

def detect_mask_opencv(image_path):
    image=cv2.imread(image_path)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        hsvFace=hsv[y:y+h, x:x+w]

        maskDetected=False
        for lower, upper in maskColors:
            mask=cv2.inRange(hsvFace, np.array(lower), np.array(upper))
            maskRatio=np.sum(mask>0)/(w*h)

            if maskRatio>0.2:
                maskDetected=True
                break

        label="Mask" if maskDetected else "No Mask"
        color=(0, 255, 0) if maskDetected else (0, 0, 255)

        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detection (Improved OpenCV)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_mask_opencv("example2.jpg") 
