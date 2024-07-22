import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self):
        # Load models and preprocessing data
        self.encoder_model = keras.models.load_model('./models/encoder_mod.h5')
        self.decoder_model = keras.models.load_model('./models/decoder_mod.h5')
        with open('./models/var.pkl', 'rb') as f:
            jj = pickle.load(f)
        self.input_token_index = jj["input_token_index"]
        self.target_token_index = jj["target_token_index"]
        self.num_decoder_tokens = jj["num_decoder_tokens"]
        self.max_encoder_seq_length = jj["max_encoder_seq_length"]
        self.num_encoder_tokens = jj["num_encoder_tokens"]
        self.reverse_target_char_index = jj["reverse_target_char_index"]
        self.max_decoder_seq_length = jj['max_decoder_seq_length']

    def decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_token_index["\t"]] = 1.0

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            states_value = [h, c]
        return decoded_sentence

    def auto_correct(self, test_text):

        encoder_test_data = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32")

        for t, char in enumerate(test_text):
            encoder_test_data[0, t, self.input_token_index[char]] = 1.0

        decoded_sentence = self.decode_sequence(encoder_test_data)
        print(test_text, '--->', decoded_sentence)
        return decoded_sentence
