import numpy as np
import math
from keras import backend as K


def fill_with_spaces(peptide, max_len):
    return peptide + ' ' * (max_len - len(peptide))


def one_hot_encode(peptide):
    aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                  'Y', ' ']

    to_one_hot = dict()
    for i, a in enumerate(aminoacids):
        v = np.zeros(len(aminoacids))
        v[i] = 1
        to_one_hot[a] = v

    result = []
    for l in peptide:
        result.append(to_one_hot[l])
    result = np.array(result)
    return np.reshape(result, (1, result.shape[0], result.shape[1]))


def one_hot_decode(peptide2Darray):
    peptide_string = ''
    for aminoacid in peptide2Darray:
        max_values = []
        max_value = max(aminoacid)
        for i, bit in enumerate(aminoacid):
            if bit == max_value:
                max_values.append(i)

        aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y', ' ']
        if len(max_values) > 1:
            element = '['
            for number in max_values:
                element = element + aminoacids[number] + ':'
            element = element + ']'
            peptide_string = peptide_string + element
        elif len(max_values) == 1:
            peptide_string = peptide_string + aminoacids[max_values[0]]
        else:
            peptide_string = peptide_string + '_'
    return peptide_string


def one_hot_encode_list(peptides):
    peptides_encoded = np.zeros(shape=(1, 21 * 50))
    for peptide in peptides:
        if (len(peptide) > 50):
            continue
        encoded_peptide = one_hot_encode(fill_with_spaces(peptide, 50))
        encoded_peptide = encoded_peptide.reshape(-1, 21 * 50)
        peptides_encoded = np.append(peptides_encoded, encoded_peptide, axis=0)

    peptides_encoded = np.delete(peptides_encoded, 0, 0)
    return peptides_encoded


def one_hot_decode_list(peptides_encoded):
    peptides_decoded = []
    for peptide_encoded in peptides_encoded:
        peptide_encoded = peptide_encoded.reshape((50, 21))
        peptide_decoded = one_hot_decode(peptide_encoded)
        peptides_decoded.append(peptide_decoded)
    return peptides_decoded


def softmax_beta(x, beta=1.5):
    x = x * beta
    return K.softmax(x)