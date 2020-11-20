import tensorflow as tf
print(tf.__version__)

with tf.device('/cpu:0'):
    x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

# 创建一个上下文环境
# 配置是将里面具体的执行过程答打印出来
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 碰到sess.run就会立即执行的运算
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()

# 在with块内部, session被设置成默认的session
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # x.initializer.run()         # 等价于 tf.get_default_session().run(x.initializer)
    # y.initializer.run()
    # result = f.eval()           # 等价于 tf.get_default_session().run(f)
    # tf.get_default_session().run(x.initializer)
    # tf.get_default_session().run(f)
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)

print(result)
