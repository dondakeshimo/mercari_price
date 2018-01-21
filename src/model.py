import sys
import time
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
        token_category = pad_sequences(token_category, maxlen=100)

        brand = self.data.brand_name
        tokenizer = Tokenizer(split="/")
        tokenizer.fit_on_texts(brand.values)
        token_brand = tokenizer.texts_to_sequences(brand.values)
        self.brand_dict = {v: k for k, v in tokenizer.word_index.items()}

        return token_category, token_brand


def main():
    start = time.time()
    file_path = sys.argv[1]
    mercari = Predict_price(file_path)
    mercari.drop_useless()
    mercari.arrange_description()
    mercari.tokenize_category_n_brand()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}[sec]".format(elapsed_time))


if __name__ == "__main__":
    main()
