# tensorflow's step

# init graph

# run graph

import tensorflow as tf

# 1. init graph
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # 묵시적으로 float32로 생성됨

# 2. run graph
sess = tf.Session()
result = sess.run([node1, node2])
print(result)