import tensorflow as tf

class Test:

    def __init__(self, data):
        self.data = data

    sess=tf.Session()    

    saver = tf.train.import_meta_graph('model/convolutional_neural_network.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))
     
        
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name("x")
    feed_dict = {x: data}
    

    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
    print(sess.run(op_to_restore,feed_dict))
