import time
import argparse
from itertools import chain
import numpy as np
import pandas as pd
from termcolor import cprint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, Model
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.layers import GRU, concatenate
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


PAD_MAXLEN = 100
TOKEN_FEAT = 10000
EMBED_DIM = 128
GRU_OUT = 1
EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 0.33
# dictionary length is 1144
# dictionary length is 4979


def load_data(file_path):
    train = pd.read_csv(file_path, sep="\t")
    test = pd.read_csv(file_path, sep="\t")
    train["target"] = np.log1p(train["price"])
    return train, test


def handle_nan(dataset):
    NO_DESC = "No description yet"
    dataset.category_name.fillna(value="None", inplace=True)
    dataset.brand_name.fillna(value="None", inplace=True)
    dataset.item_description.fillna(value="None", inplace=True)
    dataset.item_description.replace(to_replace=NO_DESC,
                                     value="None",
                                     inplace=True)
    return dataset


def tokenize_category_n_brand(self):
    category = self.train.category_name
    tokenizer = Tokenizer(split="/")
    tokenizer.fit_on_texts(category.values)
    token_category = tokenizer.texts_to_sequences(category.values)
    self.category_dict = {v: k for k, v in tokenizer.word_index.items()}
    token_category = pad_sequences(token_category, maxlen=PAD_MAXLEN)
    self.category = token_category

    brand = self.train.brand_name
    tokenizer = Tokenizer(split="\n", filters="\n")
    tokenizer.fit_on_texts(brand.values)
    token_brand = tokenizer.texts_to_sequences(brand.values)
    token_brand = np.array(list(chain.from_iterable(token_brand)))
    self.brand = token_brand
    self.brand_dict = {v: k for k, v in tokenizer.word_index.items()}


def extraction_extra_data(self):
    self.item_con = self.train.item_condition_id.values
    self.shipping = self.train.shipping.values
    self.price = self.train.price.values


def make_separate_data(self):
    sep_data = train_test_split(self.item_des, self.brand,
                                self.category, self.item_con,
                                self.shipping, self.price, test_size=0.33)
    print("training data number: {}".format(sep_data[0].shape[0]))
    print("validation data number: {}".format(sep_data[1].shape[0]))

    self.X_train = {
        "item_des": sep_data[0],
        "brand": sep_data[2],
        "category": sep_data[4],
        "item_con": sep_data[6],
        "shipping": sep_data[8],
    }
    self.Y_train = sep_data[10]
    self.X_test = {
        "item_des": sep_data[1],
        "brand": sep_data[3],
        "category": sep_data[5],
        "item_con": sep_data[7],
        "shipping": sep_data[9],
    }
    self.Y_test = sep_data[11]


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
        emb_item_con = Embedding(6, 5)(input_item_con)
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

        self.model.compile(optimizer="rmsprop",
                           loss="mean_squared_error")
        print(self.model.summary())


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


def save_model(self, checkpoint_path):
    self.model.save_weights(checkpoint_path + ".h5")
    with open(checkpoint_path + ".json", 'w') as f:
        f.write(self.model.to_json())


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return (np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean())**0.5


def argparser():
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument("-m", "--mode",
                        default=None,
                        nargs="?",
                        required=True,
                        help="train or predict")
    parser.add_argument("-i", "--input_file_path",
                        default="./data/train_sample.tsv",
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
    cprint("{:20}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed),
           "blue")
    return elapsed


def main():
    start = time.time()
    args = argparser()
    print(args)
    mercari = Predict_price(args.input_file_path)
    elapsed = time_measure("load data", start, 0)
    mercari.drop_useless()
    mercari.arrange_description()
    elapsed = time_measure("arrange des", start, elapsed)
    mercari.tokenize_category_n_brand()
    elapsed = time_measure("tokenize data", start, elapsed)
    mercari.extraction_extra_data()
    mercari.make_separate_data()
    elapsed = time_measure("complete arrange data", start, elapsed)
    mercari.make_model()
    plot_model(mercari.model, to_file="./data/model.png", show_shapes=True)
    elapsed = time_measure("make model", start, elapsed)
    mercari.train()
    elapsed = time_measure("train model", start, elapsed)
    mercari.save_model(args.checkpoint_path)
    elapsed = time_measure("save model", start, elapsed)
    K.clear_session()


if __name__ == "__main__":
    main()
