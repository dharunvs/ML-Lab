import cv2

imagePath = "pic 1.png"
imagePath = "pic 2.jpg"
cascPath = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)


image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
 gray,
#  scaleFactor=1.1,
#  minNeighbors=5,
#  minSize=(5, 5),
#  flags = cv2.FONT_HERSHEY_SIMPLEX
)
print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 print("w: " , w, "h: ", h )
cv2.imshow("Faces found", image)
cv2.waitKey(0)
