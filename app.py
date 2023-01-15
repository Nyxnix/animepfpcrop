import cv2
import sys
import os.path

def detect(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
                                     
    count = 0
    imgnorect = image.copy() # Copy image before adding rectangle
    #top_shift_scale = 0.5 # param
    #x_scale = 0.20 # param

    # All the commented out stuff in the for loop is for padding around the image, currently breaks the whole thing

    for (x, y, w, h) in faces:
        #y_shift = int(h * top_shift_scale) # Padding y axis
        #x_shift = int(w * x_scale) # Padding x axis
        face = imgnorect[y:y+h, x:x+w]
        #crop_img = imgnorect[y - y_shift:y+h, x - x_shift:x+w+x_shift] # Crop image with no rectange and add padding
        cv2.imwrite(str(count)+'.png', face) # Save face(s) to .png
        count += 1 # Incriment Counter
        imgrect = cv2.rectangle(image, (x , y), (x + w, y + h), (0, 0, 255), 2) # Add rectange to face

    # Save imgrect
    cv2.imwrite("out.png", imgrect)

if len(sys.argv) != 2:
    sys.stderr.write("usage: detect.py <filename>\n")
    sys.exit(-1)
    
detect(sys.argv[1])