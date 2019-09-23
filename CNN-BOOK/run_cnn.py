#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
# 在服务器端绘图要加上下面两句
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import warnings

warnings.filterwarnings('ignore')

# UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in
#  labels with no true samples.


base_dir = 'data/book'
# train_dir = os.path.join(base_dir, 'book_train_2010_6_4.txt')
# train_dir = os.path.join(base_dir, 'book_train_2010+2011_7_3.txt')
# train_dir = os.path.join(base_dir, 'book_train_2016_7_3.txt')
# train_dir = os.path.join(base_dir, 'book_train_2018.txt')
# train_dir = os.path.join(base_dir, 'book_train_2018-1.txt')
train_dir = os.path.join(base_dir, 'book_train_2018-2.txt')
# train_dir = os.path.join(base_dir, 'book_train_2018-3.txt')
# train_dir = os.path.join(base_dir, 'book_train_2017.txt')


# test_dir = os.path.join(base_dir, 'book_test_kj.txt')
# test_dir = os.path.join(base_dir, 'book_test_sk.txt')
# test_dir = os.path.join(base_dir, 'book_test_bd.txt')
# test_dir = os.path.join(base_dir, 'book_test_2016.txt')
# test_dir = os.path.join(base_dir, 'book_test_2018.txt')
# test_dir = os.path.join(base_dir, 'book_test_2018-1.txt')
test_dir = os.path.join(base_dir, 'book_test_2018-2.txt')
# test_dir = os.path.join(base_dir, 'book_test_2018-3.txt')
# test_dir = os.path.join(base_dir, 'book_test_2017.txt')


# val_dir = os.path.join(base_dir, 'book_val_2010_6_4.txt')
# val_dir = os.path.join(base_dir, 'book_val_2010+2011_7_3.txt')
# val_dir = os.path.join(base_dir, 'book_val_2016_7_3.txt')
# val_dir = os.path.join(base_dir, 'book_val_2018.txt')
# val_dir = os.path.join(base_dir, 'book_val_2018-1.txt')
val_dir = os.path.join(base_dir, 'book_val_2018-2.txt')
# val_dir = os.path.join(base_dir, 'book_val_2018-3.txt')
# val_dir = os.path.join(base_dir, 'book_val_2017.txt')

# vocab_dir = os.path.join(base_dir, 'book_vocab_2010.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2010+2011.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2016.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2018.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2018-1.txt')
vocab_dir = os.path.join(base_dir, 'book_vocab_2018-2.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2018-3.txt')
# vocab_dir = os.path.join(base_dir, 'book_vocab_2017.txt')


save_dir = 'checkpoints/v2_2018-2'

acc_pic_title = str(save_dir.split('/')[1])+"_Train_Val_Acc_Fig"
loss_pic_title = str(save_dir.split('/')[1])+"_Train_Val_Loss_Fig"
images_dir = 'images/'+str(save_dir.split('/')[1])
acc_pic_path = os.path.join(images_dir,acc_pic_title)
loss_pic_path = os.path.join(images_dir,loss_pic_title)

save_path = os.path.join(save_dir, 'best_validation')  # The path to save the best verification results

def get_time_dif(start_time):
    """Get used time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """evaluate the accuracy and loss of a certain data"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # Configure Tensorboard. When retraining, delete the tensorboard folder, otherwise the map will be overwritten.

    tensorboard_dir = 'tensorboard/v3_2018-2'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    global iterations

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    #  Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training:{} and validation:{} data...".format(train_dir,val_dir))
    # load train data and test data
    start_time = time.time()
    # x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    # x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)


    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # total batch
    best_acc_val = 0.0  # best accuracy of val set
    last_improved = 0  # record last improved batch
    require_improvement = 1000  # If more than 1000 rounds are not raised, end training early
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # Write the training results to tensorboard scalar every few rounds
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # Performance per round of output on training and validation sets
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                train_loss_list.append(loss_train)
                val_loss_list.append(loss_val)
                train_acc_list.append(acc_train)
                val_acc_list.append(acc_val)
                iterations += 100
                iter_list.append(iterations)

                if acc_val > best_acc_val:
                    #  record the best result
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # run optimization
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # if the correct rate of the verification set is not improved for a long time,
                # the training will be terminated early.
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(2)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # read saved model
    print("Using {}...".format(save_dir))
    print('Testing {}...'.format(test_dir))
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # save the predict result
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories,digits=4))

    # confusion matrix
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    plot_confusion_matrix(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def plotpic_loss():
    plt.figure()
    plt.plot(iter_list,train_loss_list,color='red',label='train_loss')
    plt.plot(iter_list,val_loss_list,color='blue',label='val_loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(loss_pic_title)
    plt.legend(loc='best')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plt.savefig(loss_pic_path)

def plotpic_acc():
    plt.figure()
    plt.plot(iter_list, train_acc_list, color='red', label='train_acc')
    plt.plot(iter_list, val_acc_list, color='blue', label='val_acc')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(acc_pic_title)
    plt.legend(loc='best')
    plt.savefig(acc_pic_path)

def write2txt():
    f_n = './acc_loss_data/v5_acc_loss_data.txt'
    with open(f_n, 'w') as f:
        f.write(str(iter_list))
        f.write('\n')
        f.write(str(train_loss_list))
        f.write('\n')
        f.write(str(val_loss_list))
        f.write('\n')
        f.write(str(train_acc_list))
        f.write('\n')
        f.write(str(val_acc_list))

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  #if the vocab is not exist,rebuild it
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    train_loss_list = []
    val_loss_list =[]
    train_acc_list = []
    val_acc_list = []
    iter_list = []
    iterations = -100



    if sys.argv[1] == 'train':
        train()
        plotpic_loss()
        plotpic_acc()
        # write2txt()
    else:
        test()





