#Juan
from __future__ import absolute_import, division, print_function

import random

import cv2
import numpy as np
import tensorflow as tf

from common import Sketcher

h = 200  # height
w = h*3  # width
tah = 25  # text area height


def process(img):
    # processes (resize, color, crop, etc) the image  to be sent
    # to TensorFlow model

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
    sketch = Sketcher('Super Awesome Handwritten Numbers Classifier!', [img], lambda: ((0, 255, 0), 255))

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
            print("Test")

        if (cv2.getWindowProperty('Super Awesome Handwritten Numbers Classifier!', 0) == -1) or (cv2.waitKey() == 27):
            break

    cv2.destroyAllWindows()


# CNN starts here
tf.logging.set_verbosity(tf.logging.INFO)

# application logic here

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def eval():

        # Load training and eval data
    """

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print(eval_data[1, :])
    #im = Image.fromarray(eval_data[1,:])
    #cv2.imwrite("Eval.jpg", im)

    # process(eval_data)
    """
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        #model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
        model_fn=cnn_model_fn, model_dir="/Users/Pablo Vargas/Character-Recognition/this is crazy/mnist_convnet_model")
        

    """
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    """
    # Train the model
    """
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])
    """

    # Evaluate the model and print results
    """
    evali = eval_data[0,:].reshape(28,28)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": evali},
        y=None,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.predict(input_fn=eval_input_fn)
    print(eval_results)
    output_classes = [p["classes"] for p in eval_results]
    print('the feed is ',output_classes)
    """

    return mnist_classifier

if __name__ == ("__main__"):
    draw(h, w, tah)
