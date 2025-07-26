#Training_for_caption.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical

# Load and preprocess captions
def load_captions(captions_file):
    with open(captions_file, 'r') as file:
        captions_doc = file.read()
    
    mapping = defaultdict(list)
    for line in captions_doc.strip().split('\n'):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0].split('.')[0], " ".join(tokens[1:])
        mapping[image_id].append(f"startseq {caption.lower()} endseq")
    return mapping

# Data generator to avoid memory issues
def data_generator(keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in keys:
            for caption in mapping[key]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = pad_sequences([seq[:i]], maxlen=max_length)[0], to_categorical([seq[i]], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                n += 1
                if n == batch_size:
                    yield [np.array(X1), np.array(X2)], np.array(y)
                    X1, X2, y, n = [], [], [], 0

# Train model
def train_model():
    captions_file = 'captions.txt'
    image_to_captions_mapping = load_captions(captions_file)

    # Load pre-extracted features instead of re-extracting
    with open('pickle_files/features.pkl', 'rb') as f:
        image_features = pickle.load(f)

    # Tokenization
    all_captions = [cap for caps in image_to_captions_mapping.values() for cap in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(cap.split()) for cap in all_captions)

    # Save tokenizer
    with open('pickle_files/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Split dataset
    image_ids = list(image_to_captions_mapping.keys())
    train_keys, test_keys = image_ids[:int(0.9 * len(image_ids))], image_ids[int(0.9 * len(image_ids)):]

    # Define model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train model
    batch_size = 32
    steps = len(train_keys) // batch_size
    train_gen = data_generator(train_keys, image_to_captions_mapping, image_features, tokenizer, max_length, vocab_size, batch_size)
    
    model.fit(train_gen, epochs=20, steps_per_epoch=steps, verbose=1)

    # Save trained model
    model.save('captionmodel.h5')

if __name__ == '__main__':
    train_model()
