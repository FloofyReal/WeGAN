import tensorflow as tf

a = tf.constant([[1, 3], [2, 0], [0, 1]], dtype=tf.float16)
b = tf.constant([[1, 3], [2, 1], [0, 3]], dtype=tf.float16)

print(a)

x = tf.squeeze(tf.metrics.mean_squared_error(a,b))
y = tf.squeeze(tf.metrics.mean_absolute_error(a,b))

print(x)

r = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(b ,a))))
rr = tf.reduce_mean(a)

print(r)
print(rr)

z = tf.squeeze(x)
zz = tf.squeeze(y)

print(z)

tensors = [a,b,x,y,r,rr,z,zz]

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)
    # sess.run(tf.global_variables_initializer())
    for t in tensors:
        print(sess.run(t))