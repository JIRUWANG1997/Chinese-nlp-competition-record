import tensorflow as tf
import  math
#没写placeholder,直接用numpy意思一下
length = 400
channels = 64
max_timescale = 10000.0
min_timescale = 1.0
position = tf.to_float(tf.range(length))
num_timescales = channels // 2
log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
s=tf.expand_dims(position, 1)
scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
signal = tf.reshape(signal, [1, length, channels])
sess =tf.Session()
res = sess.run(signal)
print(tf.shape(res))