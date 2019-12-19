import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    # batch_size是一个元组，所以要取出其中的第一个元素才是target的样本个数
    # print(batch_size)

    # 取出索引值
    pred = tf.math.top_k(output,maxk).indices
    # print(pred)
    pred = tf.transpose(pred,perm=[1,0])
    target_ = tf.broadcast_to(target,pred.shape)
    correct = tf.equal(pred,target_)

    res = []

    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k],[-1]),dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k/batch_size)
        res.append(acc)

    return res


if __name__ == '__main__':
    output = tf.nn.softmax(tf.random.normal([5,10]))
    target = tf.random.uniform([5],minval=1,maxval=10,dtype=tf.int32)
    # print(tf.reduce_sum(output,axis=1))
    # print(output)
    # print(target)
    res = accuracy(output,target,topk=[1,2,3,4,5,6,7,8,9,10])
    print(res)