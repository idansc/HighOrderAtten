'''
Preoricess a raw json dataset into hdf5/json files.

Caption: Use NLTK or split function to get tokens. 
'''
from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(dataset, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,ques in enumerate(dataset):
        s = ques['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)

        ques['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(dataset), i*100.0/len(dataset)) )
            sys.stdout.flush()   
    return dataset

def build_vocab_question(dataset, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for ques in dataset:
        for w in ques['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
    for ques in dataset:
        txt = ques['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        ques['final_question'] = question

    return dataset, vocab

def apply_vocab_question(dataset, wtoi):
    # apply the vocab on test.
    for ques in dataset:
        txt = ques['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt]
        ques['final_question'] = question

    return dataset

def get_top_answers(dataset, params):
    counts = {}
    for ques in dataset:
        ans = ques['ans'] 
        
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'    
    print '\n'.join(map(str,cw[:20]))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def encode_question(dataset, params, wtoi):

    max_length = params['max_length']
    N = len(dataset)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,ques in enumerate(dataset):
        question_id[question_counter] = ques['ques_id']
        label_length[question_counter] = min(max_length, len(ques['final_question'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(ques['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length, question_id


def encode_answer(dataset, atoi, num_ans):
    N = len(dataset)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, ques in enumerate(dataset):
		ans_arrays[i] = atoi.get(ques['ans'], num_ans+1) # -1 means wrong answer.
	
    return ans_arrays

def encode_mc_answer(dataset, atoi, num_ans):
    N = len(dataset)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, ques in enumerate(dataset):
        for j, ans in enumerate(ques['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, num_ans+1)
    return mc_ans_arrays

def filter_question(dataset, atoi):
    new_dataset = []
    for i, ques in enumerate(dataset):
        if ques['ans'] in atoi:
            new_dataset.append(ques)

    print 'question number reduce from %d to %d '%(len(dataset), len(new_dataset))
    return new_dataset

def get_unqiue_img(dataset):
    count_img = {}
    N = len(dataset)
    img_pos = np.zeros(N, dtype='uint32')
    ques_pos_tmp = {}
    for ques in dataset:
        count_img[ques['img_path']] = count_img.get(ques['img_path'], 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.

    for i, ques in enumerate(dataset):
        idx = imgtoi.get(ques['img_path'])
        img_pos[i] = idx

        if idx-1 not in ques_pos_tmp:
            ques_pos_tmp[idx-1] = []

        ques_pos_tmp[idx-1].append(i+1)
    
    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N,3), dtype='uint32')
    ques_pos_len = np.zeros(img_N, dtype='uint32')

    for idx, ques_list in ques_pos_tmp.iteritems():
        ques_pos_len[idx] = len(ques_list)
        for j in range(len(ques_list)):
            ques_pos[idx][j] = ques_list[j]
    return unique_img, img_pos, ques_pos, ques_pos_len

def main(params):

    # create output h5 file for training set.
    f = h5py.File(params['output_h5'], "w")

    if params['input_json']=='':
        dataset_train = json.load(open(params['input_train_json'], 'r'))
        #dataset_train = dataset_train[:5000]
        #dataset_test = dataset_test[:5000]
        # get top answers
        top_ans = get_top_answers(dataset_train, params)
        atoi = {w:i+1 for i,w in enumerate(top_ans)}
        atoi['error'] = params['num_ans']+1
        itoa = {i+1:w for i,w in enumerate(top_ans)}
        itoa[params['num_ans']+1] = 'error'
        # filter question, which isn't in the top answers.
        dataset_train = filter_question(dataset_train, atoi)

        # tokenization and preprocessing training question
        dataset_train = prepro_question(dataset_train, params)
        # create the vocab for question
        dataset_train, vocab = build_vocab_question(dataset_train, params)
        itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
        wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
        ques_train, ques_length_train, question_id_train = encode_question(dataset_train, params, wtoi)
        # get the unique image for train
        unique_img_train, img_pos_train, ques_pos_train, ques_pos_len_train = get_unqiue_img(dataset_train)
        # get the answer encoding.
        ans_train = encode_answer(dataset_train, atoi, params['num_ans'])
        MC_ans_train = encode_mc_answer(dataset_train, atoi, params['num_ans'])
        N_train = len(dataset_train)
        split_train = np.zeros(N_train)
        f.create_dataset("ques_train", dtype='uint32', data=ques_train)
        f.create_dataset("answers", dtype='uint32', data=ans_train)
        f.create_dataset("ques_id_train", dtype='uint32', data=question_id_train)
        f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
        f.create_dataset("ques_pos_train", dtype='uint32', data=ques_pos_train)
        f.create_dataset("ques_pos_len_train", dtype='uint32', data=ques_pos_len_train)
        f.create_dataset("split_train", dtype='uint32', data=split_train)
        f.create_dataset("ques_len_train", dtype='uint32', data=ques_length_train)
        f.create_dataset("MC_ans_train", dtype='uint32', data=MC_ans_train)
    else:
        loaded_train_data = json.load(open(params['input_json'], 'r'))
        itow = loaded_train_data['ix_to_word']
        wtoi = {v: k for k, v in itow.iteritems()}
        itoa = loaded_train_data['ix_to_ans']
        atoi = {v: k for k, v in itoa.iteritems()}
        unique_img_train = loaded_train_data['unique_img_train']
        
    
    dataset_test = json.load(open(params['input_test_json'], 'r'))
    # tokenization and preprocessing testing question
    dataset_test = prepro_question(dataset_test, params)
    dataset_test = apply_vocab_question(dataset_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(dataset_test, params, wtoi)

    # get the unique image for test
    unique_img_test, img_pos_test, ques_pos_test, ques_pos_len_test = get_unqiue_img(dataset_test)


    if not params['test']:
        ans_test = encode_answer(dataset_test, atoi, params['num_ans'])  #also comment line 238


    MC_ans_test = encode_mc_answer(dataset_test, atoi, params['num_ans'])

    # get the split

    N_test = len(dataset_test)
    # since the train image is already suffled, we just use the last val_num image as validation
    # train = 0, val = 1, test = 2

    #split_train[N_train - params['val_num']: N_train] = 1

    split_test = np.zeros(N_test)
    split_test[:] = 2

    
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)
    if not params['test']:
        f.create_dataset("ans_test", dtype='uint32', data=ans_test)
    f.create_dataset("ques_id_test", dtype='uint32', data=question_id_test)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)
    f.create_dataset("ques_pos_test", dtype='uint32', data=ques_pos_test)
    f.create_dataset("ques_pos_len_test", dtype='uint32', data=ques_pos_len_test)
    f.create_dataset("split_test", dtype='uint32', data=split_test)
    f.create_dataset("ques_len_test", dtype='uint32', data=ques_length_test)
    f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)

    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['uniuqe_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='vqa_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='vqa_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=3000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--input_json', default='' ,help='input existing train perprocess, usefull to process a new test file')
    parser.add_argument('--output_json', default='vqa_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='vqa_data_prepro.h5', help='output h5 file')
  
    # options
    parser.add_argument('--max_length', default=15, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    parser.add_argument('--test', default=0 ,type=int, help='token method, nltk is much more slower.')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
