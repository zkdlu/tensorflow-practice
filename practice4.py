# 모든 머신러닝 모델은 다음의 3가지 과정을 거친다.
# 1. 학습하고자 하는 가설Hypothesis h(@)를 수학적 표현식으로 나타낸다.
# 2. 가설의 성능을 측정 할 수 있는 손실 함수Loss function J(@)를 정의한다.
# 3. 손실함수 J(@)을 최소화 할 수 있는 학습 알고리즘을 설계한다.

import tensorflow as tf

# 업데이트되는 가변값에 사용 
# tf.Variable(inital_value, trainable, name)
#   initial_value: 초기값으로 shape를 포함한 상태로 지정
#   trainable: 트레이닝 가능 여부를 물으며 기본값은 True
#   name 텐서의 이름


# tf.Variable에서 초기화로 사용 할 수 있는 연산
#   tf.random_noraml: 가우시안 분포(정규 분포)에서 임의의 값을 추출
#   tf.truncated_noraml: truncated_normal 분포(끝 부분이 잘린 정규분포)
#   tf.random_uniform: 균등 분포에서 임의의 값을 추출
#   tf.constant: 특정 상수값으로 지정한 행렬
#   tf.zeros: 0 행렬
#   tf.ones: 1 행렬

#1. 변수와 플레이스홀더를 이용해 선형 회귀 모델의 그래프 구조를 정의 (Wx + b)
W = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#2. 타겟 데이터를 입력받을 플레이스 홀더로 정의
y = tf.placeholder(tf.float32)

#3. 손실 함수를 정의
loss = tf.reduce_mean(tf.square(linear_model - y))

#4. 최적화를 위한 옵티마이저 정의, 경사하강법에 학습률은 .01로 지정
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

#5. 학습을 위한 트레이닝 데이터
x_train = [1,2,3,4]
y_train = [2,4,6,8]

#6. global_variables_initializer를 이용해야 W,b를 random_normal로 했기 때문에
# W와 b가 normal distribution에서 추출한 임의의 값으로 초기화 된다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#7. feed_dict에 트레이닝 데이터를 지정하고 train_step을 실행한 후 
# 경사하강법을 수행해 파라미터를 업데이트 한다.
for i in range(10000):
    sess.run(train_step, feed_dict={x:x_train, y:y_train})

#8. 테스트 데이터를 통해 핛브이 잘 됐는지 출력 후 세션종료
x_test = [3.5, 5, 5.5, 6]

print(sess.run(linear_model, feed_dict={x:x_test}))

sess.close()