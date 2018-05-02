import cv2
import numpy as np
import tensorflow as tf

from common import Sketcher

import cnn

h = 200  # height
w = h*3  # width
tah = 25  # text area height
name = 'Super Awesome Handwritten Numbers Classifier!                              Hey, Kaur! :p'

def process(img):
    # processes (resize, color, crop, etc) the image  to be sent
    # to trained model for evaluation

    crop = img[tah:h, 0:(w//3)]   # crops
    #img2 = crop     # adjusts brightness
    resized = cv2.resize(crop, (28, 28))  # resizes image to 28*28 pixels

    inverted = invert_image(resized)

    return inverted

def invert_image(img):
    # inverts image/swaps black and white pixel values

    for i in range(28):
        for j in range(28):            
            if (img[i,j] == 1):
                img[i,j] = 0              
            elif (img[i,j] == 0):
                img[i,j] = 1
    return img

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
    text3 = "Predection"
    cv2.putText(canvas, text3, (220, 22), font3, .7, (0, 255, 0), 1)

    #right box
    text4 = "Probablities"
    cv2.putText(canvas, text4, (415, 22), font3, .7, (0, 255, 0), 1)


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

            cv2.rectangle(img,(202,tah),(400,h+tah),(255,255,255),-1)

            cv2.rectangle(img,(404,tah),(600,h+tah),(255,255,255),-1)

            imgarray = process(img)
            
            to_eval = tf.estimator.inputs.numpy_input_fn(
                x={"x": imgarray},
                num_epochs=1,
                y=None,
                shuffle=False)
                
            # eval_data = mnist.test.images  # Returns np.array
            output1 = eval().predict(input_fn=to_eval)
            output2 = eval().predict(input_fn=to_eval)

            output_classes = [c['classes'] for c in output1]
            output_prob = [p['probabilities'] for p in output2]

            probs = output_prob[0]
            vals = []

            for i in range(0, len(probs)):
                vals.append((i, probs[i]))

            sorted_probs = sorted(vals, key=lambda x: x[1])

            # print evaluation probabilities

            prob1 = str(sorted_probs[-1][0]) + ' = ' + str(sorted_probs[-1][1]) + '%'
            cv2.putText(img, prob1 , (420, 75), font3, .6, (0, 255, 0), 1)

            prob2 = str(sorted_probs[-2][0]) + ' = ' + str(sorted_probs[-2][1]) + '%'
            cv2.putText(img, prob2 , (420, 125), font3, .6, (0, 255, 0), 1)

            prob2 = str(sorted_probs[-3][0]) + ' = ' + str(sorted_probs[-3][1]) + '%'
            cv2.putText(img, prob2 , (420, 175), font3, .6, (0, 255, 0), 1)


            # print prediction
            output = str(output_classes)
            cv2.putText(img, str(output[1]), (250, 150), font4, 4, (0, 255, 0), 3)
            sketch.show()
            print("Eval")

        if (cv2.getWindowProperty(name, 0) == -1) or (cv2.waitKey() == 27):
            break

    cv2.destroyAllWindows()


# CNN starts here
tf.logging.set_verbosity(tf.logging.INFO)

def eval():

    # load trained model
    mnist_classifier = tf.estimator.Estimator(
        #model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
        model_fn=cnn.cnn_model_fn, model_dir="/Users/Pablo Vargas/Character-Recognition/mnist_convnet_model")
    
    return mnist_classifier

if __name__ == ("__main__"):
    draw(h, w, tah)
