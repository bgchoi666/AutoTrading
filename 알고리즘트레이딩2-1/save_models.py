# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# checkpoint로 저장된 train 모델들을 .h5 파일로 저장한다.

from data import config as conf
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(conf.input_size)),
        tf.keras.layers.Dense(conf.n_unit, activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 2), activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 4), activation='relu'),
        tf.keras.layers.Dense(conf.target_num, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  #callbacks=[cp-callback]
                  metrics=['accuracy'])

    checkpoint_path = "models/60M_input83_basis"

    model.save_weights(checkpoint_path)

    return model

model_pools = ["5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
          "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]
last_train = '2021-12-31'

m = create_model()

for model in model_pools:
    selected_checkpoint_path = last_train + "/60M_" + model + "_best"
    m.load_weights(selected_checkpoint_path)
    m.save("models/"+selected_checkpoint_path+",h5")