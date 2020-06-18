import tensorflow as tf
import  math
#没写placeholder

length = 400#共400个词
channels = 64#词向量维度64
max_timescale = 10000.0#mod大小
min_timescale = 1.0


position = tf.to_float(tf.range(length))#生成维度400的序列（1-400）
num_timescales = channels // 2
log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /(tf.to_float(num_timescales) - 1))#log（10000）/len(model)的一半:32
inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)#pos/10000^(index/32),index是一个序列向量
s=tf.expand_dims(position, 1)
scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)#注意高维张量乘法
signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)#前面sin,后面cos，进行直接拼接attention is all you need论文不同
signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
signal = tf.reshape(signal, [1, length, channels])#reshape回一维
sess =tf.Session()
res = sess.run(signal)
print(tf.shape(res))
