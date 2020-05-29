import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import choice

import multiprocessing

from pathlib import Path 
import io
import os
import zipfile
import tarfile
import gzip
import signal

from math import ceil
from os.path import join

import re #process text

import shutil
from functools import partial

import nltk

import torch


from torch.optim import Adam

import torch.nn as nn
import torch.utils.data

import torchtext.vocab as vocab

from torchtext.data import Field, TabularDataset, Dataset
from torchtext.data.example import Example
from torchtext.utils import download_from_url
from torchtext.utils import unicode_csv_reader

from sklearn.metrics.pairwise import cosine_similarity

from model import *
from loss import *
from data import *

import time
from sys import float_info, stdout


def _tokenize_str(str_): #preprocess text
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_) #rajoute des espace autour des parentheses
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()

def _run(dataset,
         data_generator,
         num_batches,
         vocabulary_size,
         context_size,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver_is_dbow):

    if model_ver_is_dbow:
        model = DBOW(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)
    else:
        model = DM(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents.".format(len(dataset)))
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")

    best_loss = float("inf")
    prev_model_file_path = None

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []

        for batch_i in range(num_batches):
            batch = next(data_generator)
            if torch.cuda.is_available():
                batch.cuda_()

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)
            else:
                x = model.forward(
                    batch.context_ids,
                    batch.doc_ids,
                    batch.target_noise_ids)

            x = cost_func.forward(x)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer.step()
            _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        # prev_model_file_path = save_training_state(
        #     data_file_name,
        #     model_ver,
        #     vec_combine_method,
        #     context_size,
        #     num_noise_words,
        #     vec_dim,
        #     batch_size,
        #     lr,
        #     epoch_i,
        #     loss,
        #     state,
        #     save_all,
        #     generate_plot,
        #     is_best_loss,
        #     prev_model_file_path,
        #     model_ver_is_dbow)
        print('loss', loss, 'best_loss', best_loss)


        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))
    return model


def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end='')
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()

def _runemb(path,dataset,
         data_generator,
         num_batches,
         text_field,
         #vocabulary_size,
         context_size,
         #vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver_is_dbow):


    
    model = model = DMemb(len(dataset), text_field, path)
    vocabulary_size = torch.tensor(len(text_field.vocab.stoi))
    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents.".format(len(dataset)))
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")

    best_loss = float("inf")
    prev_model_file_path = None

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []

        for batch_i in range(num_batches):
            batch = next(data_generator)
            if torch.cuda.is_available():
                batch.cuda_()

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)
            else:
                x = model.forward(
                    batch.context_ids,
                    batch.doc_ids,
                    batch.target_noise_ids)

            x = cost_func.forward(x)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer.step()
            _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        # prev_model_file_path = save_training_state(
        #     data_file_name,
        #     model_ver,
        #     vec_combine_method,
        #     context_size,
        #     num_noise_words,
        #     vec_dim,
        #     batch_size,
        #     lr,
        #     epoch_i,
        #     loss,
        #     state,
        #     save_all,
        #     generate_plot,
        #     is_best_loss,
        #     prev_model_file_path,
        #     model_ver_is_dbow)
        print('loss', loss, 'best_loss', best_loss)


        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))
    return model