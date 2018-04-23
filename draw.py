import random

import cv2
import numpy

from common import Sketcher

h = 200 # height
w = 200 # width
tah = 25 # text area height

def process(img):
    # processes (resize, color, crop, etc) the image  to be sent 
    # to TensorFlow model
    
    crop = img[tah:h, 0:w]   # crops
    img2 = crop * 255        # adjusts brightness
    resized = cv2.resize(img2, (28, 28)) # resizes image to 28*28 pixels
    
    cv2.imwrite('output.jpg', resized)

    return resized

def main():
    
    canvas = numpy.ones((h + tah, w))

    cv2.line(canvas, (0, tah-1), (w, tah-1), (0, 0, 0), 1)
    cv2.line(canvas, (145,0), (145, tah-1), (.5, 255, 255), 1)

    font = cv2.FONT_HERSHEY_PLAIN
    font2 = cv2.FONT_HERSHEY_TRIPLEX
    text1 = "Press 'C' to clear"
    cv2.putText(canvas, text1, (5, 10), font, 0.9, (0.5, 255, 0), 1)
    text2 = "Press 'T' to test"
    cv2.putText(canvas, text2, (5, 22), font, 0.9, (0.5, 255, 0), 1)

    img = canvas.copy()
    sketch = Sketcher('Draw', [img], lambda : ((0, 255, 0), 255))    

    while (True):

        if (cv2.waitKey() == ord('c')):
            print("Clear")
            img[:] = canvas
            sketch.show()

        if (cv2.waitKey() == ord('t')):
            
            process(img)
            output = str(random.randint(0, 10))
            cv2.putText(img, output, (162, 22), font2, 1, (0, 255, 0), 1)
            sketch.show()
            print("Test")
              
        if (cv2.getWindowProperty('Draw', 0) == -1) or (cv2.waitKey() == 27):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
