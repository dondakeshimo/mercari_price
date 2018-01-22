import sys
import time
import argparse
from itertools import chain
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, Model
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.layers import GRU, concatenate
# from keras.utils import plot_model
# from keras import backend as K
from sklearn.model_selection import train_test_split


PAD_MAXLEN = 100
TOKEN_FEAT = 10000
EMBED_DIM = 128
GRU_OUT = 1
EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 0.33
# dictionary length is 1144
# dictionary length is 4979


class Predict_price():
    def __init__(self, file_path):
        self.data = pd.read_table(file_path)

    def arrange_description(self):
        NO_DES = "No description yet"
        item_des = self.data.item_description
        item_des = item_des.fillna(NO_DES)
        item_des = item_des.apply(lambda x: 0 if x == NO_DES else 1)
        self.data.loc[:, "item_des"] = item_des
        self.data = self.data.drop("item_description", axis=1)

    def drop_useless(self):
        self.data = self.data.drop("name", axis=1)
        self.data = self.data.fillna("None")

    def tokenize_category_n_brand(self):
        category = self.data.category_name
        tokenizer = Tokenizer(split="/")
        tokenizer.fit_on_texts(category.values)
        token_category = tokenizer.texts_to_sequences(category.values)
        self.category_dict = {v: k for k, v in tokenizer.word_index.items()}
        token_category = pad_sequences(token_category, maxlen=PAD_MAXLEN)

        brand = self.data.brand_name
        tokenizer = Tokenizer(split="\n", filters="\n")
        tokenizer.fit_on_texts(brand.values)
        token_brand = tokenizer.texts_to_sequences(brand.values)
        token_brand = np.array(list(chain.from_iterable(token_brand)))
        self.brand_dict = {v: k for k, v in tokenizer.word_index.items()}

        return token_category, token_brand

    def make_model(self, pre_trained_model_path=None):
        if pre_trained_model_path:
            with open(pre_trained_model_path + ".json", "rt")as f:
                json_model = f.read()
            self.model = model_from_json(json_model)
            self.model.compile(loss="mean_squared_error",
                               optimizer="adam",
                               metrics=["accuracy"])
            self.model.load_weights(pre_trained_model_path + ".h5")
            print(self.model.summary())
        else:
            input_item_des = Input(shape=[1], name="item_des")
            input_brand = Input(shape=[1], name="brand")
            input_category = Input(shape=[PAD_MAXLEN], name="category")
            input_item_con = Input(shape=[1], name="item_con")
            input_shipping = Input(shape=[1], name="shipping")

            emb_item_des = Embedding(2, 5)(input_item_des)
            emb_brand = Embedding(TOKEN_FEAT, 30)(input_brand)
            emb_category = Embedding(TOKEN_FEAT, 30)(input_category)
            emb_item_con = Embedding(2, 5)(input_item_con)
            emb_shipping = Embedding(2, 5)(input_shipping)

            rnn_layer = GRU(8)(emb_category)

            main_layer = concatenate([Flatten()(emb_item_des),
                                      Flatten()(emb_brand),
                                      Flatten()(emb_item_con),
                                      Flatten()(emb_shipping),
                                      rnn_layer])

            temp_dense = Dense(512, activation="relu")(main_layer)
            main_layer = Dropout(0.3)(temp_dense)
            temp_dense = Dense(96, activation="relu")(main_layer)
            main_layer = Dropout(0.2)(temp_dense)

            output = Dense(1, activation="linear")(main_layer)

            self.model = Model(input=[input_item_des, input_brand,
                                      input_category, input_item_con,
                                      input_shipping],
                               output=[output])
            print(self.model.summary())

    def separate_data(self):
        sep_data = train_test_split(self.X, self.Y, test_size=0.33)
        self.X_train, self.X_test, self.Y_train, self.Y_test = sep_data
        print("training data number: {}".format(self.X_train.shape[0]))
        print("validation data number: {}".format(self.X_test.shape[0]))

    def train(self):
        self.model.fit(self.X_train,
                       self.Y_train,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=0.1,
                       verbose=1)

    def evaluate(self):
        print("Evaluation")
        return self.model.evaluate(self.X_test,
                                   self.Y_test,
                                   batch_size=BATCH_SIZE,
                                   verbose=1)

    def predict(self, target):
        print("Prediction")
        return self.model.predict(target, verbose=1)

    def predict_significant_terms(self, target):
        prediction = self.predict(target)
        significant_terms = []

        for i in range(len(self.X)):
            tmp = self.X[i][prediction[i] >= 0.5]
            significant_terms.append([])
            for t in tmp:
                significant_terms[i].append(self.word_dict[t])
            significant_terms[i] = list(set(significant_terms[i]))
            significant_terms[i].sort()

        return significant_terms

    def save_model(self, checkpoint_path):
        self.model.save_weights(checkpoint_path + ".h5")
        with open(checkpoint_path + ".json", 'w') as f:
            f.write(self.model.to_json())

    def count_accuracy(self, target, prediction):
        total_target = 0
        predicted_target = 0
        not_significant_terms = 0
        for i in target.index:
            set_target = set(target[i])
            set_pred = set(prediction[i])
            total_target += len(set_target)
            predicted_target += len(set_target & set_pred)
            not_significant_terms += len(set_pred - set_target)
        print("count_accuracy: {}".format(predicted_target / total_target))
        print("mistake_count: {}".format(not_significant_terms))
        return total_target, predicted_target, not_significant_terms


def argparser():
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument("-m", "--mode",
                        default=None,
                        nargs="?",
                        required=True,
                        help="train or predict")
    parser.add_argument("-i", "--input_file_path",
                        default="./data/issues(5).csv",
                        nargs="?",
                        help="input data path")
    parser.add_argument("-p", "--pre_trained_model_path",
                        default=None,
                        nargs="?",
                        help="to load checkpoint h5 file path")
    parser.add_argument("-c", "--checkpoint_path",
                        default="./data/save_test",
                        nargs="?",
                        help="checkpoint h5 file path")
    return parser.parse_args()


def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    print("{:20}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed))
    return elapsed


def main():
    start = time.time()
    file_path = sys.argv[1]
    mercari = Predict_price(file_path)
    elapsed = time_measure("load data", start, 0)
    mercari.drop_useless()
    mercari.arrange_description()
    elapsed = time_measure("arrange data", start, elapsed)
    mercari.tokenize_category_n_brand()
    elapsed = time_measure("tokenize data", start, elapsed)
    # plot_model(mercari.model, to_file="./data/model.png", show_shapes=True)
    # K.clear_session()


if __name__ == "__main__":
    main()
