import itertools
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, GRU, Dropout, Flatten, concatenate
from keras.layers import Dense
from keras.models import Model


TRAIN_DATA = "../data/train.tsv"
SAMPLE_DATA = "../data/sample_submission.csv"
NO_DES = "No description yet"

train = pd.read_csv(TRAIN_DATA, sep="\t")
train.head()

item_des = train.item_description
train.loc[:, "item_des"] = item_des.apply(lambda x: 0 if x == NO_DES else 1)
train = train.drop("item_description", axis=1)
train = train.drop("name", axis=1)
train = train.fillna("None")
train.head()

category = train.category_name
tokenizer = Tokenizer(split="/")
tokenizer.fit_on_texts(category.values)
token_cat = tokenizer.texts_to_sequences(category.values)
token_cat = pad_sequences(token_cat, maxlen=100)
token_cat
cat_dict = {v: k for k, v in tokenizer.word_index.items()}

tokenizer = Tokenizer(split="\n", filters="\n")
tokenizer.fit_on_texts(train.brand_name.values)
token_brand = tokenizer.texts_to_sequences(train.brand_name.values)
len(token_brand)
len(list(itertools.chain.from_iterable(token_brand)))
token_brand = np.array(list(itertools.chain.from_iterable(token_brand)))
token_brand.shape
brand_dict = {v: k for k, v in tokenizer.word_index.items()}

input_item_des = Input(shape=[1], name="item_des")
input_brand = Input(shape=[1], name="brand")
input_category = Input(shape=[100], name="category")
input_item_con = Input(shape=[1], name="item_con")
input_shipping = Input(shape=[1], name="shipping")

emb_item_des = Embedding(2, 5)(input_item_des)
emb_brand = Embedding(10000, 30)(input_brand)
emb_category = Embedding(10000, 30)(input_category)
emb_item_con = Embedding(6, 5)(input_item_con)

rnn_layer = GRU(8)(emb_category)

main_l = concatenate([Flatten()(emb_item_des),
                      Flatten()(emb_brand),
                      Flatten()(emb_item_con),
                      rnn_layer])

temp_dense = Dense(512, activation="relu")(main_l)
main_l = Dropout(0.3)(temp_dense)
temp_dense = Dense(96, activation="relu")(main_l)
main_l = Dropout(0.2)(temp_dense)

output = Dense(1, activation="linear")(main_l)

model = Model(input=[input_item_des, input_brand, input_category,
                     input_item_con, input_shipping],
              output=[output])

model.compile(optimizer="rmsprop",
              loss="mean_squared_error")

model.fit({
              "item_des": train.item_des.values,
              "brand": token_brand,
              "category": token_cat,
              "item_con": train.item_condition_id.values,
              "shipping": train.shipping.values
          }, train.price.values, epochs=1, batch_size=32)

train.shipping.values
token_brand.shape == train.item_condition_id.shape
token_brand
train.item_condition_id[train.item_condition_id > 5]


def a(b):
    def c(d):
        print(d)
    return c


a = a(1)(2)
