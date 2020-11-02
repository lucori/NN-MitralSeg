import time
import argparse
from segment.rnmf_segment import SegRNMF
from segment.nnmf_segment import SegNNMF
import torch
from utils import get_free_gpu
import os
import numpy as np
import json
from parser import ConfigParserEcho
from shutil import copyfile
import csv
import socket
from echos import DataMaster
import pickle5 as pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    # current time for file names
    date_time = time.strftime("%Y%m%d-%H%M%S")
    print("Time:", date_time)

    # check if gpu is available
    if torch.cuda.is_available():
        device = 'cuda:' + str(get_free_gpu())
    else:
        device = 'cpu'
    print('INFO: Start on device %s' % device)
    print(os.path.join(dir_path, '../configuration/NNMF.ini'))
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config', default=os.path.join(dir_path, 'configuration/NNMF.ini'))
    args = parser.parse_args()
    conf = args.config
    config = ConfigParserEcho()
    config.read(conf)

    fact_type = config['Parameters']['fact_type']
    epochs = config['Parameters'].get('epochs', 0)
    n_steps = config['Parameters'].get('n_steps', 0)
    batch_size = config['Parameters']['batch_size']
    learning_rate = config['Parameters']['learning_rate']
    mlp_size = config['Parameters']['mlp_size']
    gmf_size = config['Parameters']['gmf_size']
    l1_mult = config['Parameters']['l1_mult']
    l21_mult = config['Parameters']['l21_mult']
    embedding_mult = config['Parameters']['embedding_mult']
    spat_temp_mult = config['Parameters']['spat_temp_mult']
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
    threshold_mv = float(config['Parameters']['threshold_mv'])
    threshold_wd = float(config['Parameters']['threshold_wd'])

    if socket.gethostname() == "isegpu2":
        num_workers = max(16, int(num_workers))

    patients = json.loads(config['Load_Save']['patients'])

    if train_test_split == 1: train_test_split = None

    if load_raw_data:
        print("Loading raw data.")
        dtMaster = DataMaster(config)
        dtMaster.load()
        # chek echos loaded succesfully
        assert len(dtMaster.dt.echos) > 0, "No ECHO loaded!"
        data_folder = dtMaster.dt.echos[0].pickle_folder

    # get list of the echos in the data folder
    video_list = os.listdir(os.path.join(dir_path, str(data_folder)))
    if files_done:
        video_list = [p for p in video_list if all([a not in p for a in files_done])]
    if patients:
        video_list = [p for p in video_list if any([a in p for a in patients])]
    print(video_list)

    scores = []
    for i in range(len(video_list)):

        patient_id = video_list[i].split('.')[0]
        print("\nSegmenting valve for patient: {} ({}/{})".format(patient_id, i+1, len(video_list)))

        #dt = load_zipped_pickle(os.path.join(dir_path, str(data_folder), video_list[i]))
        with open(os.path.join(dir_path, str(data_folder), video_list[i]), 'rb') as f:
            dt = pickle.load(f)

        x = dt.matrix3d
        x = np.nan_to_num(x)

        if str(fact_type) == 'nnmf':

            seg = SegNNMF(l1_mult=float(l1_mult), l21_mult=float(l21_mult), embedding_mult=float(embedding_mult),
                          epochs=int(epochs), n_steps=int(n_steps), learning_rate=float(learning_rate), mlp_size=int(mlp_size),
                          gmf_size=int(gmf_size), batchsize=int(batch_size), num_workers=int(num_workers), device=device,
                          embedding_nmf_init=bool(embedding_nmf_init), gmf_net_init=gmf_net_init, mlp_layers=mlp_layers,
                          threshold_layers=threshold_layers, window_size=window_size, train_test_split=train_test_split,
                          patience=patience, min_delta=min_delta, early_stopping=early_stopping,
                          spat_temp_mult=float(spat_temp_mult), save_data_every=save_data_every,
                          save_tensorboard_summary_every=save_tensorboard_summary_every,
                          search_window_size=search_window_size, opt_flow_window_size=float(opt_flow_window_size),
                          connected_struct=connected_struct, morph_op=morph_op, option=option, threshold_mv=threshold_mv,
                          threshold_wd=threshold_wd)
        else:
            seg = SegRNMF(rank=2, sparsity_coef=(0.2, 0.001), window_size=window_size,
                          search_window_size=search_window_size, opt_flow_window_size=float(opt_flow_window_size),
                          option=option, max_iter=20, thresh1=99, thresh2=99.2,
                          time_series_masking=time_series_masking, threshold_wd=threshold_wd)

        seg.set_labels(dt.labels)
        seg.set_x(x)
        seg.set_save_location(os.path.join(date_time, patient_id))
        seg.set_labels(dt.labels)

        save_location_runs = os.path.join("runs", date_time, patient_id)
        os.makedirs(save_location_runs)

        # train the model
        score = seg.train(save_location=save_location_runs)
        print(score)
        scores.append(score)

        del seg
        time.sleep(10)

    # copy config
    copyfile(conf, os.path.join("runs", date_time, "config.ini"))

    # write scores to *.csv
    keys = scores[0].keys()
    f_score = os.path.join("runs", date_time, 'scores.csv')
    with open(f_score, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(scores)

    # evaluate scores:
    print("\nValve IOU: ", np.mean([e['iou'] for e in scores]))
    print("Valve Dice", np.mean([e['dice'] for e in scores]))
    print("Window Acc", np.mean([e['window_acc'] for e in scores]))
    print("Window IOU", np.mean([e['window_iou'] for e in scores]))

    print("Number window over 0.65: ", len(
        [e['window_acc'] for e in scores if round(e['window_acc'], 2) >= 0.65]))
    print("Number window over 0.85: ", len(
        [e['window_acc'] for e in scores if round(e['window_acc'], 2) >= 0.85]))

    print("Saved scores to: ", f_score)
    print("Saved segmentation to: out/SegRNMF/original/runs/{}".format(date_time))
    print("DONE")

