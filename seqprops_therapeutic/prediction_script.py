if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    import models
    from seqprops import SequentialPropertiesEncoder


    sequences = ["AAC", "CAC"]          # Sekvence za koje se radi predvidanje
    feature_set = ['Hydrophobicity_Aboderin', 'tScales_T3', 'Hydrophobicity_Fauchere', 'stScales_ST4', 'VHSE_VHSE3', 'BLOSUM_BLOSUM5', 'Hydrophobicity_Ponnuswamy', 'crucianiProperties_PP1', 'Hydrophobicity_Welling', 'Hydrophobicity_Chothia']
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True, max_seq_len=46, selected_properties=feature_set)
    encoded_sequences = encoder.encode(sequences)
    model = models.create_seq_model(input_shape=encoded_sequences.shape[1:], conv1_filters=64, conv2_filters=64, conv_kernel_size=4, num_cells=128, dropout=0.1)

    # Copy weights from pretrained model to the new model
    loaded_model = tf.keras.models.load_model("pretrained_models/amp")
    model.set_weights(loaded_model.get_weights())

    # Predvidanje i ispis predvidanja
    y_pred = model.predict(encoded_sequences, verbose=0)
    for idx in range(len(y_pred)):
        predicted_probability = y_pred[idx][0]
        if predicted_probability > 0.5:
            print(True)
        else:
            print(False)