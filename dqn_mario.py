from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
USE_MOVEMENT = RIGHT_ONLY

import collections
import random
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 输出 ERROR + FATAL
assert tf.__version__.startswith('2.')

learning_rate= 0.0002
gamma = 0.99
buffer_limit = 5000
batch_size = 32

class ReplayBuffer(object):
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        # 从回放池采样n个5元组
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        # 按类别进行整理
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        # 转换成Tensor
        return tf.constant(s_lst, dtype=tf.float32), \
               tf.constant(a_lst, dtype=tf.int32), \
               tf.constant(r_lst, dtype=tf.float32), \
               tf.constant(s_prime_lst, dtype=tf.float32), \
               tf.constant(done_mask_lst, dtype=tf.float32)

    def size(self):
        return len(self.buffer)

class QNetWork(keras.Model):
    def __init__(self):
        super(QNetWork, self).__init__()
        self.network = Sequential([  # 网络容器
            layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层, 6个3x3卷积核
            layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
            layers.ReLU(),  # 激活函数
            layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16个3x3卷积核
            layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
            layers.ReLU(),  # 激活函数
            layers.Flatten(),  # 打平层，方便全连接层处理

            layers.Dense(120, activation='relu'),  # 全连接层，120个节点
            layers.Dense(84, activation='relu'),  # 全连接层，84节点
            layers.Dense(len(USE_MOVEMENT))  # 全连接层，10个节点
        ])

    def call(self, x, training=None):

        x = self.network(x)
        return x

    def sample_action(self, s, epsilon):
        s = tf.constant(s, dtype=tf.float32)
        # 一维变二维
        s = tf.expand_dims(s, axis=0)
        out = self(s)[0]
        coin = random.random()
        if coin < epsilon:
            return random.choice(range(len(USE_MOVEMENT)))
        else:
            return int(tf.argmax(out))

def train(q, q_target, memory, optimizer):
    huber = losses.Huber()
    for i in range(10):
        s, a, r, s_prime, done = memory.sample(batch_size)
        with tf.GradientTape() as tape:
            q_out = q(s)
            indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
            indices = tf.concat([indices, a], axis=1)
            q_a = tf.gather_nd(q_out, indices)  # 根据indices提取对应位置的元素
            q_a = tf.expand_dims(q_a, axis=1)
            max_q_prime = tf.reduce_max(q_target(s_prime), axis=1, keepdims=True)
            target = r + gamma*max_q_prime*done
            loss = huber(q_a, target)
        grads = tape.gradient(loss, q.trainable_variables)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = JoypadSpace(env, USE_MOVEMENT)
    interval = 20
    q = QNetWork()
    q_target = QNetWork()
    input_shape = (batch_size, 240, 256, 3)
    q.build(input_shape=input_shape)
    q_target.build(input_shape=input_shape)
    for src, dest in zip(q.variables, q_target.variables):
        dest.assign(src)
    memory = ReplayBuffer()

    score = 0.
    optimizer = optimizers.Adam(lr=learning_rate)
    for n_epi in range(10000):
        eqsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s = env.reset()
        for t in range(10000):
            a = q.sample_action(s, eqsilon)
            s_prime, r, done, _ = env.step(a)
            env.render()
            done = 0. if done else 1.
            memory.put((s, a, r, s_prime, done))
            s = s_prime
            score += r
            if not done:
                break
        print ("epeide :   {} ".format(n_epi))
        if memory.size() > 100:
            train(q, q_target, memory, optimizer)
        # print("22,  ", tf.size(q), tf.size(q))
        if n_epi % interval == 0 and n_epi != 0:
            # print(q.variables, q_target.variables)
            for src, dest in zip(q.variables, q_target.variables):
                dest.assign(src)  # 影子网络权值来自Q
            print(" # of epsode {}, avg_score {}, buffer size {}".format(n_epi, score/interval, memory.size()))
            score = 0.
        if n_epi % 200 == 0 and not n_epi:
            q_target.network.save_weights('dqn_weights{}.ckpt'.format(int(n_epi / 200)))
    env.close()



if __name__ == "__main__":
    main()
