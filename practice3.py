import tensorflow as tf

# constant와 달리 임의의 값을 입력받아 출력 가능
# tf.placeholder(dtype, shpae, name)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  #묵시적으로 tf.add(a,b)로 정의 됨


with tf.Session() as sess : # C#의 using 문
    print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
    print(sess.run(adder_node, feed_dict={a:[1,3],b:[2,4]}))

adder_node_triple = adder_node * 3 # tf.multiply(adder_node, 3)

with tf.Session() as sess:
    print(sess.run(adder_node_triple, feed_dict={a:3, b:4.5}))

