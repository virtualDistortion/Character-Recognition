#Pablo
import cv2
import numpy as np
import tensorflow as tf

from scipy import ndimage

import cnn
from common import Sketcher

h = 200  # height
w = h*3  # width
tah = 25  # text area height
name = 'Super Awesome Handwritten Numbers Classifier!                              Hey, Kaur! :p'

# load trained model
classifier = tf.estimator.Estimator(
model_fn=cnn.cnn_model_fn, model_dir="D:/OneDrive/2018 Spring Artificial Intelligence/Character-recognition/mnist_convnet_model")
#model_fn=cnn.cnn_model_fn, model_dir="/Users/Pablo Vargas/Character-Recognition/mnist_convnet_model")

def process(img):
    # processes (resize, color, crop, etc) the image  to be sent
    # to trained model for evaluation

    crop = img[tah:h, 0:(w//3)]   # crops
    
    resized = cv2.resize(crop, (28, 28))  # resizes image to 28*28 pixels

    
    resized = ndimage.gaussian_filter(resized,0.25) #gassian filter with sigma = 0.25
    ho = ndimage.sobel(resized,0) #horizontal sobel edge detection
    v = ndimage.sobel(resized,1) # vertical sobel edge detection
    inverted = np.hypot(ho,v) # combine both edge detection images    

    return inverted #return processed image

#defines drawing space
def draw(h, w, tah):
    
    canvas = np.ones((h + tah, w), dtype=np.float32)

    cv2.line(canvas, (0, tah-1), (w, tah-1), (0, 0, 0), 1)
    cv2.line(canvas, (201, 0), (201, h+tah), (.5, 255, 255), 1)
    cv2.line(canvas, (402, 0), (402, h+tah), (.5, 255, 255), 1)

    font  = cv2.FONT_HERSHEY_PLAIN
    font2 = cv2.FONT_HERSHEY_TRIPLEX
    font3 = cv2.FONT_HERSHEY_SIMPLEX
    font4 = cv2.FONT_HERSHEY_DUPLEX

    #left box
    text1 = "Press 'C' to clear"
    cv2.putText(canvas, text1, (20, 10), font, 0.9, (0.5, 255, 0), 1)
    text2 = "Press 'E' to evaluate"
    cv2.putText(canvas, text2, (20, 22), font, 0.9, (0.5, 255, 0), 1)

    #middle box
    text3 = "Prediction"
    cv2.putText(canvas, text3, (240, 22), font3, .7, (0, 255, 0), 1)

    #right box
    text4 = "Confidence"
    cv2.putText(canvas, text4, (440, 22), font3, .7, (0, 255, 0), 1)

    img = canvas.copy()
    sketch = Sketcher(name, [img], lambda: ((0, 255, 0), 255))

    while (True):

        if cv2.waitKey(0) == ord('c'):
            # clear the screen
            
            print("Clear")
            img[:] = canvas
            sketch.show()

        if cv2.waitKey(0) == ord('e'):
            # evaluate drawing
            
            # remove previous results from screen
            cv2.rectangle(img,(202,tah),(400,h+tah),(255,255,255),-1)
            cv2.rectangle(img,(404,tah),(600,h+tah),(255,255,255),-1)

            img2 = process(img) #call process img to apply filtering and edge detection
            
            #convert image into tensor input
            to_eval = tf.estimator.inputs.numpy_input_fn(
                x={"x": img2}, y=None, shuffle=False)
                
            # evaluate input image,  Returns np.array
            output1 = classifier.predict(input_fn=to_eval)
            output2 = classifier.predict(input_fn=to_eval)

            #extract prediction class and probabilities from classfier
            output_classes = [c['classes'] for c in output1]
            output_prob = [p['probabilities'] for p in output2]

            #sort probabilities
            probs = output_prob[0]
            vals = []

            for i in range(0, len(probs)):
                vals.append((i, probs[i]))

            sorted_probs = sorted(vals, key=lambda x: x[1])
            
            #determine confidency level
            if sorted_probs[-1][1] >= 0.9:
                conf = 'Very Confident'
            elif sorted_probs[-1][1] < 0.9 and sorted_probs[-1][1] >= 0.8:
                conf = 'Confident'
            elif sorted_probs[-1][1] < 0.8 and sorted_probs[-1][1] >= 0.65:
                conf = 'Somewhat Confident'
            elif sorted_probs[-1][1] < 0.65 :
                conf = 'Not Confident'

            # print evaluation confidence
            prob2 = str(sorted_probs[-2][0]) + ' = ' + str(sorted_probs[-2][1])
            cv2.putText(img, conf , (430, 115), font3, .6, (0, 255, 0), 1)

            
            # print prediction
            output = str(output_classes)
            cv2.putText(img, str(output[1]), (250, 150), font4, 4, (0, 255, 0), 3)
            sketch.show()
            print("Eval")

        if (cv2.getWindowProperty(name, 0) == -1) or (cv2.waitKey() == 27):
            break

    cv2.destroyAllWindows()

if __name__ == ("__main__"):
    draw(h, w, tah)
