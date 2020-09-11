import tensorflow as tf
import matplotlib.pyplot as plt
for example in tf.python_io.tf_record_iterator("testduo.tfrecords"):
    img = tf.train.Example.FromString(example).features.feature['image_raw'].bytes_list.value[0]
    img = tf.decode_raw(img, tf.uint8)
    
    label = tf.train.Example.FromString(example).features.feature['label'].int64_list.value[0]

    with tf.Session() as sess:
        img = sess.run(img).reshape([2, 96,96])
        print(img)
        print(label)
        plt.imshow(img[0])
        plt.show()
        plt.imshow(img[1])
        plt.show()
        
    