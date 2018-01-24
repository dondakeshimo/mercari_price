import time
import argparse
import numpy as np
import pandas as pd
from termcolor import cprint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, Model
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.layers import GRU, concatenate
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


EPOCHS = 2
BATCH_SIZE = 512 * 3


class Predict_price():
    def __init__(self, dir_path):
        self.train = pd.read_csv(dir_path + "train.tsv", sep="\t")
        self.test = pd.read_csv(dir_path + "test.tsv", sep="\t")
        self.train["target"] = np.log1p(self.train["price"])

    def handle_nan(self, dataset):
        NO_DESC = "No description yet"
        dataset.category_name.fillna(value="None", inplace=True)
        dataset.brand_name.fillna(value="None", inplace=True)
        dataset.item_description.fillna(value="None", inplace=True)
        dataset.item_description.replace(to_replace=NO_DESC,
                                         value="None",
                                         inplace=True)
        return dataset

    def handle_nan_process(self):
        self.train = self.handle_nan(self.train)
        self.test = self.handle_nan(self.test)

    def label_encode(self):
        le = LabelEncoder()

        le.fit(np.hstack([self.train.category_name, self.test.category_name]))
        self.train['category'] = le.transform(self.train.category_name)
        self.test['category'] = le.transform(self.test.category_name)

        le.fit(np.hstack([self.train.brand_name, self.test.brand_name]))
        self.train['brand_name'] = le.transform(self.train.brand_name)
        self.test['brand_name'] = le.transform(self.test.brand_name)
        del le

    def tokenize_seq_data(self):
        raw_text = np.hstack([self.train.category_name.str.lower(),
                              self.train.item_description.str.lower(),
                              self.train.name.str.lower()])

        tok_raw = Tokenizer()
        tok_raw.fit_on_texts(raw_text)
        self.train["seq_category_name"] = tok_raw.texts_to_sequences(
            self.train.category_name.str.lower())
        self.test["seq_category_name"] = tok_raw.texts_to_sequences(
            self.test.category_name.str.lower())
        self.train["seq_item_desc"] = tok_raw.texts_to_sequences(
            self.train.item_description.str.lower())
        self.test["seq_item_desc"] = tok_raw.texts_to_sequences(
           self.test.item_description.str.lower())
        self.train["seq_name"] = tok_raw.texts_to_sequences(
            self.train.name.str.lower())
        self.test["seq_name"] = tok_raw.texts_to_sequences(
            self.test.name.str.lower())

    def search_max_len(self):
        max_train_name = np.max(self.train.seq_name.apply(lambda x: len(x)))
        max_test_name = np.max(self.test.seq_name.apply(lambda x: len(x)))
        self.max_name_seq = np.max([max_train_name, max_test_name])
        max_train_desc = np.max(
            self.train.seq_item_desc.apply(lambda x: len(x)))
        max_test_desc = np.max(
            self.test.seq_item_desc.apply(lambda x: len(x)))
        self.max_item_desc_seq = np.max([max_train_desc, max_test_desc])

    def define_max(self):
        self.MAX_NAME_SEQ = 15
        self.MAX_ITEM_DESC_SEQ = 65
        self.MAX_CATEGORY_NAME_SEQ = 20
        self.MAX_TEXT = np.max([np.max(self.train.seq_name.max()),
                                np.max(self.test.seq_name.max()),
                                np.max(self.train.seq_category_name.max()),
                                np.max(self.test.seq_category_name.max()),
                                np.max(self.train.seq_item_desc.max()),
                                np.max(self.test.seq_item_desc.max())]) + 2
        self.MAX_CATEGORY = np.max([self.train.category.max(),
                                    self.test.category.max()]) + 1
        self.MAX_BRAND = np.max([self.train.brand_name.max(),
                                 self.test.brand_name.max()]) + 1
        self.MAX_CONDITION = np.max([self.train.item_condition_id.max(),
                                     self.test.item_condition_id.max()]) + 1

    def arrange_target(self):
        self.train["target"] = np.log(self.train.price + 1)
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train["target"] = self.target_scaler.fit_transform(
            self.train.target.values.reshape(-1, 1))

    def get_keras_data(self, dataset):
        name = pad_sequences(
            dataset.seq_name, maxlen=self.MAX_NAME_SEQ)
        item_desc = pad_sequences(
            dataset.seq_item_desc, maxlen=self.MAX_ITEM_DESC_SEQ)
        category_name = pad_sequences(
            dataset.seq_category_name, maxlen=self.MAX_CATEGORY_NAME_SEQ)
        X = {
            "name": name,
            "item_desc": item_desc,
            "brand_name": np.array(dataset.brand_name),
            "category": np.array(dataset.category),
            "category_name": category_name,
            "item_cond": np.array(dataset.item_condition_id),
            "num_vars": np.array(dataset.shipping)
        }
        return X

    def get_keras_data_process(self):
        self.X_train = self.get_keras_data(self.train)
        self.X_test = self.get_keras_data(self.test)

    def make_model(self, pre_trained_model_path=None):
        if pre_trained_model_path:
            with open(pre_trained_model_path + ".json", "rt")as f:
                json_model = f.read()
            self.model = model_from_json(json_model)
            self.model.compile(loss="mean_squared_error",
                               optimizer="adam",
                               metrics=["accuracy"])
            self.model.load_weights(pre_trained_model_path + ".h5")
            # print(self.model.summary())
        else:
            name = Input(shape=[self.X_train["name"].shape[1]],
                         name="name")
            item_desc = Input(shape=[self.X_train["item_desc"].shape[1]],
                              name="item_desc")
            brand_name = Input(shape=[1],
                               name="brand_name")
            category = Input(shape=[1],
                             name="category")
            ctgr_name = Input(shape=[self.X_train["category_name"].shape[1]],
                              name="category_name")
            item_cond = Input(shape=[1],
                              name="item_cond")
            num_vars = Input(shape=[1],
                             name="num_vars")

            emb_name = Embedding(self.MAX_TEXT, 60)(name)
            emb_item_desc = Embedding(self.MAX_TEXT, 60)(item_desc)
            emb_category_name = Embedding(self.MAX_TEXT, 20)(ctgr_name)
            emb_brand_name = Embedding(self.MAX_BRAND, 10)(brand_name)
            emb_category = Embedding(self.MAX_CATEGORY, 10)(category)
            emb_item_cond = Embedding(self.MAX_CONDITION, 5)(item_cond)

            rnn_layer1 = GRU(16)(emb_item_desc)
            rnn_layer2 = GRU(16)(emb_category_name)
            rnn_layer3 = GRU(8)(emb_name)

            main_layer = concatenate([Flatten()(emb_brand_name),
                                      Flatten()(emb_category),
                                      Flatten()(emb_item_cond),
                                      rnn_layer1,
                                      rnn_layer2,
                                      rnn_layer3,
                                      num_vars])

            temp_dense = Dense(512, activation="relu")(main_layer)
            main_layer = Dropout(0.05)(temp_dense)
            temp_dense = Dense(16, activation="relu")(main_layer)
            main_layer = Dropout(0.05)(temp_dense)

            output = Dense(1, activation="linear")(main_layer)

            self.model = Model(inputs=[name, item_desc, brand_name,
                                       category, ctgr_name,
                                       item_cond, num_vars],
                               outputs=[output])

            self.model.compile(optimizer="adam",
                               loss="mse",
                               metrics=["mae", rmsle_cust])
            # print(self.model.summary())

    def train_model(self):
        self.model.fit(self.X_train,
                       self.train.target.values,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=0.1,
                       verbose=1)

    def predict(self, input_data):
        pred = self.model.predict(input_data, verbose=1, batch_size=BATCH_SIZE)
        pred = self.target_scaler.inverse_transform(pred)
        pred = np.exp(pred) - 1
        return pred

    def save_model(self, checkpoint_path):
        self.model.save_weights(checkpoint_path + ".h5")
        with open(checkpoint_path + ".json", "w") as f:
            f.write(self.model.to_json())


def get_callbacks(checkpoint_path, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(checkpoint_path, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return (np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean())**0.5


def argparser():
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument("-m", "--mode",
                        default=None,
                        nargs="?",
                        help="train or predict")
    parser.add_argument("-i", "--input_dir_path",
                        default="./data/",
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
    cprint("{:22}: {:15.2f}[sec]{:15.2f}[sec]".format(section, lap, elapsed),
           "blue")
    return elapsed


def main():
    start = time.time()
    args = argparser()
    print(args)
    mercari = Predict_price(args.input_dir_path)
    elapsed = time_measure("load data", start, 0)
    mercari.handle_nan_process()
    elapsed = time_measure("handle nan", start, elapsed)
    mercari.label_encode()
    elapsed = time_measure("label encode data", start, elapsed)
    mercari.tokenize_seq_data()
    elapsed = time_measure("tokenize data", start, elapsed)
    mercari.define_max()
    mercari.arrange_target()
    mercari.get_keras_data_process()
    elapsed = time_measure("complete arrange data", start, elapsed)
    mercari.make_model()
    plot_model(mercari.model, to_file="./data/model.png", show_shapes=True)
    elapsed = time_measure("make model", start, elapsed)
    mercari.train_model()
    elapsed = time_measure("train model", start, elapsed)
    mercari.save_model(args.checkpoint_path)
    elapsed = time_measure("save model", start, elapsed)
    K.clear_session()


if __name__ == "__main__":
    main()
