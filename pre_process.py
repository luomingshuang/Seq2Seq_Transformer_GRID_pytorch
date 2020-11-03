import os
import pickle

from tqdm import tqdm

from options import pickle_file, grid_wav, grid_text, grid_images
from utils import ensure_folder

import glob
import numpy as np

letters = ['<sos>', '<eos>', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#char_list = [' ', '!', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^']

def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB

    wav_files = glob.glob(os.path.join(grid_wav, split, '*', '*', '*', '*.wav'))
    #print(wav_files)
    #print(wav_files[:10])
    
    samples = []
    for wav_file in wav_files:
        items = wav_file.split(os.path.sep)
        text = os.path.join(grid_text, split, items[-4], 'align', items[-1][:-3] + 'align')
        images = os.path.join(grid_images, split, items[-4], 'video', items[-2], items[-1][:-4])
        with open(text, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        txt = ' '.join(txt).upper()
        #trn = [letters.index('<sos>')]
        #trn = list(txt) + ['<eos>']
        #print(trn)
        trn = list(txt)
        trn = [letters.index(c) for c in trn]

        #trn = txt.split(' ')
        #print(trn)
        # for c in (trn):
        #     build_vocab(c)
        # trn = [VOCAB[c] for c in trn]

        #print(trn)
        # for i in range(35 - len(trn)):
        #     trn.append(1)
        
        print({'trn':trn, 'wave':wav_file, 'images':images})
        samples.append({'trn':trn, 'wave':wav_file, 'images':images})
        # print(trn)
        # print(text)
        # print(items)
    print('split: {}, num_files: {}'.format(split, len(samples)))
    #print(samples)
    return samples

def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token
    # with open(tran_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()

if __name__ == "__main__":
    # VOCAB = {'<sos>': 0, '<eos>': 1}
    # IVOCAB = {0: '<sos>', 1: '<eos>'}
    data = dict()
    # data['VOCAB'] = VOCAB
    # data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['val'] = get_data('val')


    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_val: ' + str(len(data['val'])))
