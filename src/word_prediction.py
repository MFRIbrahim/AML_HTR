import numpy as np
from beam_search import ctcBeamSearch


class SimpleWordDecoder(object):
    def __init__(self, idx_to_char):
        self.__idx_to_char = idx_to_char

    def __call__(self, matrix, join=True):
        # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
        char_matrix = np.argmax(matrix, axis=2)
        col_count, row_count = char_matrix.shape

        words = list()
        for row in range(row_count):
            word = [self.__idx_to_char[char_matrix[col][row]] for col in range(col_count)]
            if join:
                word = "".join(word)
            words.append(word)

        return words


class BestPathDecoder(SimpleWordDecoder):
    def __init__(self, idx_to_char):
        super().__init__(idx_to_char)

    def __call__(self, matrix, join=True):
        # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
        output = super().__call__(matrix, join=False)

        # clean the output, i.e. remove multiple letters not seperated by '|' and '|'
        last_letter = "abc"  # invalid label
        current_letter = ""
        output_clean = []
        for i in range(len(output)):
            sub = []
            for j in range(len(output[i])):
                current_letter = output[i][j]
                if output[i][j] != "|" and output[i][j] != last_letter:
                    sub.append(output[i][j])
                last_letter = current_letter
            output_clean.append(sub)

        if join:
            for i in range(len(output_clean)):
                output_clean[i] = "".join(output_clean[i]).strip()
        # print(output)
        return output_clean


class BeamDecoder(object):
    def __init__(self, char_list, beam_width=4):
        self.__char_list = char_list
        self.__beam_width = beam_width

    def __call__(self, matrix):
        return ctcBeamSearch(matrix, "".join(self.__char_list), None, beamWidth=self.__beam_width)