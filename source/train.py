import time
import argparse
from segment.rnmf_segment import SegRNMF
from segment.nnmf_segment import SegNNMF
import torch
import pickle
from utils import get_free_gpu
import os
import numpy as np
import json
from parser import ConfigParserEcho
from shutil import copyfile
import csv
import socket

from echos import DataMaster

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    # current time for file names
    time = time.strftime("%Y%m%d-%H%M%S")
    print("Time:", time)

    # check if gpu is available
    if torch.cuda.is_available():
        device = 'cuda:' + str(get_free_gpu())
    else:
        device = 'cpu'
    print('INFO: Start on device %s' % device)
    print(os.path.join(dir_path, '../configuration/test_config.ini'))
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config', default=os.path.join(dir_path, '../configuration/test_config.ini'))
    args = parser.parse_args()
    conf = args.config
    config = ConfigParserEcho()
    config.read(conf)

    fact_type = config['Parameters']['fact_type']
    epochs = config['Parameters']['epochs']
    batch_size = config['Parameters']['batch_size']
    learning_rate = config['Parameters']['learning_rate']
    mlp_size = config['Parameters']['mlp_size']
    gmf_size = config['Parameters']['gmf_size']
    l1_mult = config['Parameters']['l1_mult']
    l21_mult = config['Parameters']['l21_mult']
    embedding_mult = config['Parameters']['embedding_mult']
    embedding_nmf_init = config.getboolean('Parameters', 'embedding_nmf_init')
    gmf_net_init = config.getboolean('Parameters', 'gmf_net_init')
    data_folder = config['Parameters']['data_folder']
    mlp_layers = json.loads(config['Parameters']['mlp_layers'])
    threshold_layers = json.loads(config['Parameters']['threshold_layers'])
    window_size = json.loads(config['Parameters']['window_size'])
    search_window_size = json.loads(config['Parameters']['search_window_size'])
    opt_flow_window_size = config['Parameters']['opt_flow_window_size']
    connected_struct = config.getboolean('Parameters', 'connected_struct')
    morph_op = config.getboolean('Parameters', 'morph_op')

    load_raw_data = config['Load_Save']['load_raw_data'].lower() == 'true'
    train_test_split = json.loads(config['Parameters']['train_test_split'])
    patience = json.loads(config['Parameters']['patience'])
    min_delta = json.loads(config['Parameters']['min_delta'])
    early_stopping = config.getboolean('Parameters', 'early_stopping')
    num_workers = json.loads(config['Parameters']['num_workers'])
    save_data_every = json.loads(config['Parameters']['save_data_every'])
    save_tensorboard_summary_every = json.loads(config['Parameters']['save_tensorboard_summary_every'])
    files_done = json.loads(config['Load_Save']['files_done'])
    option = config['Parameters']['option']
    time_series_masking = config.getboolean('Parameters', 'time_series_masking')

    if socket.gethostname() == "isegpu2":
        num_workers = max(16, int(num_workers))

    patients = json.loads(config['Load_Save']['patients'])

    if train_test_split == 1: train_test_split=None

    if load_raw_data:
        print("Loading raw data.")
        dtMaster = DataMaster(config)
        dtMaster.load()

    # get list of the echos in the data folder
    patient_list = os.listdir(os.path.join(dir_path, str(data_folder)))
    print(patients)
    if files_done:
        patient_list = [p for p in patient_list if all([a not in p for a in files_done])]
    if patients:
        patient_list = [p for p in patient_list if any([a in p for a in patients])]
    scores = []
    print(patient_list)
    for i in range(len(patient_list)):

        patient_id = patient_list[i].split('.')[0]
        print("Segmenting valve for patient: ", patient_id)

        with open(os.path.join(dir_path, str(data_folder)) + patient_list[i], 'rb') as f:
            dt = pickle.load(f)

        x = dt.matrix3d
        x = np.nan_to_num(x)

        if str(fact_type) == 'nnmf':

            seg = SegNNMF(l1_mult=float(l1_mult), l21_mult=float(l21_mult), embedding_mult=float(embedding_mult),
                          epochs=int(epochs), learning_rate=float(learning_rate), mlp_size=int(mlp_size),
                          gmf_size=int(gmf_size), batchsize=int(batch_size), num_workers=int(num_workers), device=device,
                          embedding_nmf_init=bool(embedding_nmf_init), gmf_net_init=gmf_net_init, mlp_layers=mlp_layers,
                          threshold_layers=threshold_layers, window_size=window_size, train_test_split=train_test_split,
                          patience=patience, min_delta=min_delta, early_stopping=early_stopping,
                          save_data_every=save_data_every, save_tensorboard_summary_every=save_tensorboard_summary_every,
                          search_window_size=search_window_size, opt_flow_window_size=float(opt_flow_window_size),
                          connected_struct=connected_struct, morph_op=morph_op)
        else:
            seg = SegRNMF(rank=2, sparsity_coef=(0.2, 0.001), search_window_size=search_window_size,
                          opt_flow_window_size=float(opt_flow_window_size), option=option, max_iter=20,
                          thresh1=99, thresh2=99.2, time_series_masking=time_series_masking)

        seg.set_x(x)
        seg.set_save_location(os.path.join(time, patient_id))
        seg.set_labels(dt.labels)

        save_location_runs = os.path.join("runs", time, patient_id)
        os.makedirs(save_location_runs)
        copyfile(conf, os.path.join(save_location_runs, "config.ini"))

        score = seg.train(save_location=save_location_runs)
        scores.append(score)

        del seg

    # write scores to *.csv
    keys = scores[0].keys()
    with open(os.path.join("runs", time, 'scores.csv'), 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(scores)
