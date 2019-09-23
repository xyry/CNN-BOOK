from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

base_dir = 'data/book'
vocab_dir = os.path.join(base_dir, 'book_vocab_2018-2.txt')

save_dir = 'checkpoints/v3_2018-2'
save_path = os.path.join(save_dir, 'best_validation')  # Best verification result saved path


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # Read the saved model
        self.out = list()


    def predict(self, content):
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls,self.out = self.session.run([self.model.y_pred_cls,self.model.prob], feed_dict=feed_dict)
        # print(self.out)
        # print(self.out[0,0],self.out[0,1])
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    data_list = list()
    # test_file_path = './file_process/test_booklist_1_500_mix.txt'
    # test_file_path = './file_process/test_booklist_2_251_not_buy.txt'
    test_file_path = './file_process/test_booklist_3_249_buy.txt'
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(str(line).split('\n')[0])
    for content in data_list:
        res = cnn_model.predict(content)
        print("The probability of buying this book is %.3f%%"%((cnn_model.out[0,0])*100))

