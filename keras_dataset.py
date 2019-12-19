import tensorflow as tf
from tensorflow.keras.datasets import mnist,cifar10,cifar100,fashion_mnist


def prepare_mnist_features_and_labels(x,y):
    # numpy通常为float64格式，TensorFlow通常为float32格式
    x = tf.cast(x,dtype=tf.float32)/255.0
    y = tf.cast(y,dtype=tf.int64)
    return x,y


def mnist_dataset():
    # 读取数据为numpy格式
    (x,y),(x_val,y_val) = fashion_mnist.load_data()
    y = tf.one_hot(y,depth=10)
    y_val = tf.one_hot(y_val,depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    ds_val = ds.map(prepare_mnist_features_and_labels)
    ds_val = ds.shuffle(10000).batch(100)

    return ds,ds_val


if __name__ == '__main__':
    # 1.读取数据为numpy格式
    # x,y,x_test,y_test是numpy格式
    (x,y),(x_test,y_test) = mnist.load_data()

    print('x-shape:',x.shape,'y-shape:',y.shape)
    print('x-min:',x.min(),'x-max:',x.max(),'x-mean:',x.mean())

    (x,y),(x_test,y_test) = fashion_mnist.load_data()

    print('x-shape:',x.shape,'y-shape:',y.shape)
    print('x-min:',x.min(),'x-max:',x.max(),'x-mean:',x.mean())

    # (x,y),(x_test,y_test) = cifar10.load_data()

    # print('x-shape:',x.shape,'y-shape:',y.shape)
    # print('x-min:',x.min(),'x-max:',x.max(),'x-mean:',x.mean())

    # (x,y),(x_test,y_test) = cifar100.load_data()

    # print('x-shape:',x.shape,'y-shape:',y.shape)
    # print('x-min:',x.min(),'x-max:',x.max(),'x-mean:',x.mean())

    # 2.将numpy格式数据 → tensor格式并构建迭代器
    db = tf.data.Dataset.from_tensor_slices(x)
    it = iter(db)
    print(next(it).shape)

    db = tf.data.Dataset.from_tensor_slices((x,y))
    it = iter(db)
    # 取出image的shape
    print(next(it)[0].shape)

    # 3.完整例子
    ds,ds_val = mnist_dataset()
    print(next(iter(ds))[0].shape)



