"""
@author: Guansong Pang

Source code for the PReNet algorithm in KDD'23.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, concatenate, Dropout
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import regularizers

import matplotlib.pyplot as plt
import argparse

import time

from scipy.sparse import vstack, csc_matrix
from sklearn.model_selection import train_test_split
from utils_new import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file, dataLoading_noheader
from data_interpolation import inject_noise, inject_noise_sparse

MAX_INT = np.iinfo(np.int32).max


data_format = 0

uu=0; au=4; aa=8
#uu=0; au=1; aa=2

ensemble_size = 30

h_lambda = 0.1


def regression_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def pair_generator(x, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:        
        if data_format == 0:
            samples1, samples2, training_labels = pair_batch_generation(x, outlier_indices, inlier_indices, Y, batch_size, rng)
        else:
            samples1, samples2, training_labels = pair_batch_generation_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield([samples1, samples2], training_labels)
        if (counter > nb_batch):
            counter = 0
 
def pair_batch_generation(x_train, outlier_indices, inlier_indices, Y, batch_size, rng):
    '''batchs of samples.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    pairs1 = np.empty((batch_size, dim))  
    pairs2 = np.empty((batch_size, dim))  
    labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    block_size = int(batch_size / 4)
    sid = rng.choice(n_inliers, block_size*4, replace = False)
    pairs1[0:block_size*2] = x_train[inlier_indices[sid[0:block_size*2]]]
    pairs2[0:block_size*2] = x_train[inlier_indices[sid[block_size*2:block_size*4]]]
    labels += 2*block_size*[uu]
    
    sid = rng.choice(n_inliers, block_size, replace = False)
    pairs1[block_size*2:block_size*3] = x_train[inlier_indices[sid]]
    sid = rng.choice(n_outliers, block_size)
    pairs2[block_size*2:block_size*3] = x_train[outlier_indices[sid]]
    labels += block_size*[au]
    
    for i in np.arange(block_size*3, batch_size): 
        sid = rng.choice(n_outliers, 2, replace = False)
        z1 = x_train[outlier_indices[sid[0]]]
        z2 = x_train[outlier_indices[sid[1]]]
        pairs1[i] = z1
        pairs2[i] = z2
        labels += [aa]

    return pairs1, pairs2,  np.array(labels).astype(float)

def pair_batch_generation_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''batchs of samples.
    Alternates between positive and negative pairs.
    '''      
    pairs1 = np.empty((batch_size))  
    pairs2 = np.empty((batch_size))  
    labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    j = 0
    for i in range(batch_size):    
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 2, replace = False)
            z1 = inlier_indices[sid[0]]
            z2 = inlier_indices[sid[1]]
            pairs1[i] = z1
            pairs2[i] = z2                
            labels += [uu]
        else:
            if j % 2 == 0:
                sid = rng.choice(n_inliers, 1)
                z1 = inlier_indices[sid]
                sid = rng.choice(n_outliers, 1)
                z2 = outlier_indices[sid]
                pairs1[i] = z1
                pairs2[i] = z2
                labels += [au]

            else:
                sid = rng.choice(n_outliers, 2, replace = False)
                z1 = outlier_indices[sid[0]]
                z2 = outlier_indices[sid[1]]
                pairs1[i] = z1
                pairs2[i] = z2
                labels += [aa]
            j += 1
    pairs1 = x_train[pairs1, :].toarray()
    pairs2 = x_train[pairs2, :].toarray()
    return pairs1, pairs2,  np.array(labels).astype(float)

def reg_network_deeper(input_shape):

    x_input = Input(shape=input_shape)    
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(h_lambda), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(h_lambda), name = 'hl2')(intermediate)    
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(h_lambda), name = 'hl3')(intermediate)
    base_network = Model(x_input, intermediate)
    print(base_network.summary())  
    

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    input_merge = concatenate([processed_a, processed_b])
    anomaly_score = Dense(1, activation='linear',  name = 'score')(input_merge)    
    
    
    model = Model([input_a, input_b], anomaly_score)
#    print(model.summary())
    
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model

def reg_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(h_lambda), name = 'hl1')(x_input)
    base_network = Model(x_input, intermediate)
#    print(base_network.summary())  
    

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    input_merge = concatenate([processed_a, processed_b])
    # input_merge = Dense(20, activation='relu',  name = 'interaction')(input_merge)
    anomaly_score = Dense(1, activation='linear',  name = 'score')(input_merge)    
    
    
    model = Model([input_a, input_b], anomaly_score)
#    print(model.summary())
    
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model

def reg_network_no_feature_learner(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''    

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    input_merge = concatenate([input_a, input_b])
    anomaly_score = Dense(1, activation='linear',  name = 'score')(input_merge)    
    
    
    model = Model([input_a, input_b], anomaly_score)
    print(model.summary())
    
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=regression_loss, optimizer=rms)
    return model

def load_model_weight_predict(model_name, input_shape, network_depth, x_test, inliers, outliers):
    if network_depth == 2:
        model = reg_network(input_shape)
    elif network_depth == 1:
        model = reg_network_no_feature_learner(input_shape)
    else:
        model = reg_network_deeper(input_shape)    
    model.load_weights(model_name)
    scoring = Model(inputs=model.input, outputs=model.output)    
    
    if data_format == 0:
        runs = ensemble_size
        rng = np.random.RandomState(42) 
        test_size = x_test.shape[0]
        scores = np.zeros((test_size, runs))
        n_sample = inliers.shape[0]
        for i in np.arange(runs):
            idx = rng.choice(n_sample, 1)
            obj = inliers[idx]            
            inlier_seed = np.tile(obj, (test_size, 1))
            scores[:, i] = scoring.predict([inlier_seed, x_test]).flatten()
        mean_score = np.mean(scores, axis = 1) 
        
        runs = ensemble_size
        rng = np.random.RandomState(42) 
        test_size = x_test.shape[0]
        scores = np.zeros((test_size, runs))
        n_sample = outliers.shape[0]
        for i in np.arange(runs):
            idx = rng.choice(n_sample, 1)
            obj = outliers[idx]            
            outlier_seed = np.tile(obj, (test_size, 1))
            scores[:, i] = scoring.predict([x_test, outlier_seed]).flatten()
        mean_score += np.mean(scores, axis = 1)         
        
        scores = mean_score / 2
    else:
        data_size = x_test.shape[0]
        count = 512
        if count > data_size:
            count = data_size      
        
        runs = ensemble_size
        scores_a = np.zeros((data_size, runs))
        scores_u = np.zeros((data_size, runs))
        
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            rng = np.random.RandomState(42) 
            n_sample = inliers.shape[0]
            for j in np.arange(runs):
                idx = rng.choice(n_sample, 1)
                obj = inliers[idx].toarray()          
                inlier_seed = np.tile(obj, (count - i, 1))
                scores_u[i:count, j] = scoring.predict([inlier_seed, subset]).flatten()

            rng = np.random.RandomState(42) 
            n_sample = outliers.shape[0]
            for j in np.arange(runs):
                idx = rng.choice(n_sample, 1)
                obj = outliers[idx].toarray()            
                outlier_seed = np.tile(obj, (count - i, 1))
                scores_a[i:count, j] = scoring.predict([subset, outlier_seed]).flatten()
            
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
                
        assert count == data_size        
        mean_score = np.mean(scores_u, axis = 1) 
        mean_score += np.mean(scores_a, axis = 1) 
        scores = mean_score / 2
        
    return scores

    
def run_prenet(args):
    names = args.data_set.split(',')
    network_depth = int(args.network_depth)
#    names = ['kddcup99_normalized_sklearn_8_anomalies_small']
#    names = ['UNSW_NB15_traintest_backdoor']
#    names = ['UNSW_NB15_196feat_Shellcode_3000']
#    names = ['UNSW_NB15_196feat_Generic_3000']
#    names = ['UNSW_NB15_196feat_Fuzzers_3000']
#    names = ['probe_u2r']
#    names = ['kddcup99_r2l']
#    names = ['bank-additional-full_normalised']
#    names = ['UNSW_NB15_traintest_analysis']    
#    names = ['UNSW_NB15_traintest_Fuzzers', 'UNSW_NB15_traintest_Reconnaissance', 'UNSW_NB15_traintest_DoS']
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)  
        filename = nm.strip()
        n_outliers = 0
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()    
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]  
        n_outliers_org = outliers.shape[0]  
        train_time = 0
        test_time = 0
        global h_lambda
        h_lambda = float(args.h_lambda)
        global uu, au, aa
        ordinal_labels = args.ordinal_labels
        ordinal_labels = ordinal_labels.split(',')
        uu = float(ordinal_labels[0]); au = float(ordinal_labels[1]); aa = float(ordinal_labels[2])        
        global ensemble_size
        ensemble_size = args.ensemble_size
        
        for i in np.arange(runs):  
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            n_noise  = len(np.where(labels == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)                
            
            rng = np.random.RandomState(42) 
            if data_format == 0:                
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                
                noises = inject_noise(outliers, n_noise)
                x_train = np.append(x_train, noises, axis = 0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]                               
                
                noises = inject_noise_sparse(outliers, n_noise)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            n_samples_trn = x_train.shape[0]
            input_shape = x_train.shape[1:]
            n_outliers = len(outlier_indices)               
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            Y = np.zeros(x_train.shape[0])
            Y[outlier_indices] = 1
         
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size    
            nb_batch = args.nb_batch  
            
            if network_depth == 2:
                model = reg_network(input_shape)
            elif network_depth == 1:
                model = reg_network_no_feature_learner(input_shape)
            else:
                model = reg_network_deeper(input_shape)
            
            start_time = time.time() 
            model_name = "./model/prenet_" + filename + "_" + str(args.cont_rate) + "cr_"  + str(args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True)
            history = model.fit_generator(pair_generator(x_train, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng),
                                steps_per_epoch = nb_batch,
                                epochs = epochs,
                                callbacks=[checkpointer])            
#            plt.figure(figsize=(5, 5))
#            plt.plot(history.history['loss'])
#            plt.grid()
#            plt.title('model loss')
#            plt.xlabel('epochs')
#            plt.ylabel('loss')
#            plt.show()
            train_time += time.time() - start_time                
                
            start_time = time.time() 
            
            scores = load_model_weight_predict(model_name, input_shape, network_depth,
                                               x_test,  x_train[inlier_indices], x_train[outlier_indices]) 
            rauc[i], ap[i] = aucPerformance(scores, y_test)   
       
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time/runs
        test_time = test_time/runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
        print("average runtime: %.4f seconds" % (train_time + test_time))
        ordinal_labels = str(h_lambda)+":"+ str(uu) + "_" + str(au) + "_" + str(aa)
        writeResults(ordinal_labels + "_"+filename+'_'+str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)

def run_prenet_unseenanomaly(args):
#    test_list = ['backdoor']
#    train_list = ['Generic']
#    test_list = ['analysis']
#    train_list = ['backdoor', 'Fuzzers', 'backdoor_Fuzzers']
#    test_list = ['backdoor', 'Generic', 'Fuzzers']
#    train_list = ['backdoor', 'Generic', 'Fuzzers', 'backdoor_Generic', 'Generic_Fuzzers', 'backdoor_Fuzzers']

    test_list = ['backdoor', 'DoS', 'Fuzzers', 'Reconnaissance']
    train_list = ['backdoor', 'DoS', 'Fuzzers', 'Reconnaissance', \
                  'Reconnaissance_DoS_Fuzzers', 'DoS_backdoor_Fuzzers', 'Reconnaissance_backdoor_Fuzzers',\
                  'Reconnaissance_backdoor_DoS', 'Reconnaissance_DoS', 'Reconnaissance_Fuzzers', \
                  'Reconnaissance_backdoor', 'DoS_backdoor', 'DoS_Fuzzers', 'backdoor_Fuzzers']
    network_depth = int(args.network_depth)
    for nm in test_list:
        for nm2 in train_list:
            if (nm == nm2) or (nm in nm2):
                continue
            filename = 'UNSW_NB15_traintest_'+nm2
            runs = args.runs
            rauc = np.zeros(runs)
            ap = np.zeros(runs)  
            global data_format
            data_format = int(args.data_format)
            if data_format == 0:
                x, labels = dataLoading(args.input_path + filename + ".csv")
            else:
                x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
                x = x.tocsr()    
            outlier_indices = np.where(labels == 1)[0]
            outliers = x[outlier_indices]  
            n_outliers_org = outliers.shape[0]   
            
            train_time = 0
            test_time = 0
            for i in np.arange(runs):  
                x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                print(filename + ': round ' + str(i))
                outlier_indices = np.where(y_train == 1)[0]
                n_outliers = len(outlier_indices)
                print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
                
                n_noise  = len(np.where(labels == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
                n_noise = int(n_noise)                
                
                rng = np.random.RandomState(42) 
                if data_format == 0:                
                    if n_outliers > args.known_outliers:
                        mn = n_outliers - args.known_outliers
                        remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                        x_train = np.delete(x_train, remove_idx, axis=0)
                        y_train = np.delete(y_train, remove_idx, axis=0)
                    
                    noises = inject_noise(outliers, n_noise)
                    x_train = np.append(x_train, noises, axis = 0)
                    y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
                
                else:
                    if n_outliers > args.known_outliers:
                        mn = n_outliers - args.known_outliers
                        remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                        retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                        retain_idx = list(retain_idx)
                        x_train = x_train[retain_idx]
                        y_train = y_train[retain_idx]                               
                    
                    noises = inject_noise_sparse(outliers, n_noise)
                    x_train = vstack([x_train, noises])
                    y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
                
                outlier_indices = np.where(y_train == 1)[0]
                inlier_indices = np.where(y_train == 0)[0]
                print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
                n_samples_trn = x_train.shape[0]
                input_shape = x_train.shape[1:]
                n_outliers = len(outlier_indices)               
                print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
                Y = np.zeros(x_train.shape[0])
                Y[outlier_indices] = 1
                # print(Y)
                input_shape = x_train.shape[1:]
                epochs = args.epochs
                batch_size = args.batch_size    
                nb_batch = args.nb_batch  
                
                if network_depth == 2:
                    model = reg_network(input_shape)
                elif network_depth == 1:
                    model = reg_network_no_feature_learner(input_shape)
                else:
                    model = reg_network_deeper(input_shape)
                
                start_time = time.time() 
                model_name = "./model/prenet_" + filename + "_" + str(args.cont_rate) + "cr_"  + str(args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5"
                checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                               save_best_only = True, save_weights_only = True)
                history = model.fit_generator(pair_generator(x_train, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng),
                                    steps_per_epoch = nb_batch,
                                    epochs = epochs,
                                    callbacks=[checkpointer])            
    #            plt.figure(figsize=(5, 5))
    #            plt.plot(history.history['loss'])
    #            plt.grid()
    #            plt.title('model loss')
    #            plt.xlabel('epochs')
    #            plt.ylabel('loss')
    #            plt.show()
                train_time += time.time() - start_time                
                    
                start_time = time.time() 
                
                
                print(x_test.shape)
                outlier_indices = np.where(y_test == 1)[0]
                inlier_indices_train = np.where(y_train == 0)[0]
                outlier_indices_train = np.where(y_train == 1)[0]
    #            print(outlier_indices.shape)
                x_test = np.delete(x_test, outlier_indices, axis=0)
                y_test = np.delete(y_test, outlier_indices, axis=0)
                new_anomalies = dataLoading_noheader(args.input_path+nm+'_anomalies_only.csv')
                x_test = np.append(new_anomalies,x_test, axis = 0)
                y_test = np.append(np.ones((new_anomalies.shape[0], 1)), y_test)
                scores = load_model_weight_predict(model_name, input_shape, network_depth,
                                               x_test,  x_train[inlier_indices_train], x_train[outlier_indices_train]) 
                rauc[i], ap[i] = aucPerformance(scores, y_test)       
                
#
#                plt.figure(i)
#                outlier_indices = np.where(y_test == 1)[0]
#                plt.plot(np.arange(len(outlier_indices),scores.shape[0]), scores[len(outlier_indices):scores.shape[0]], 'bo', linewidth=2, markersize=8)
#                plt.plot(outlier_indices, scores[outlier_indices], 'r+', linewidth=2, markersize=12)
#                plt.xlabel('id')
#                plt.ylabel('score')
            test_name=nm2+'->>'+nm
            print(test_name)
            mean_auc = np.mean(rauc)
            std_auc = np.std(rauc)
            mean_aucpr = np.mean(ap)
            std_aucpr = np.std(ap)
            train_time = train_time/runs
            test_time = test_time/runs
            print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
            print("average runtime: %.4f seconds" % (train_time + test_time))
            writeResults(test_name+'_'+str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                         network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=60, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--ensemble_size", type=int, default=1, help="ensemble_size. Using a size of one runs much faster while being able to obtain similarly good performance as using a size of 30.")
parser.add_argument("--h_lambda", type=float, default=0.01, help="regularization parameter")
parser.add_argument("--ordinal_labels", type=str, default="0,4,8", help="regularization parameter")
parser.add_argument("--input_path", type=str, default='data/', help="the path of the data sets")
# parser.add_argument("--data_set", type=str, default='KDD2014_donors_10feat_nomissing_normalised, census-income-full-mixed-binarized, \
#                     creditcardfraud_normalised, celeba_baldvsnonbald_normalised, UNSW_NB15_traintest_DoS, UNSW_NB15_traintest_Reconnaissance,\
#                     UNSW_NB15_traintest_Fuzzers, UNSW_NB15_traintest_Backdoor, w7a-libsvm-nonsparse_normalised,\
#                     bank-additional-full_normalised, annthyroid_21feat_normalised\
#                     ', help="a list of data set names")
parser.add_argument("--data_set", type=str, default='news20_5Per_Otl', help="a list of data set names")                    
parser.add_argument("--data_format", choices=['0','1'], default='1',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./results/prenet_0.02contrate_2depth_10runs_lambda'+str(h_lambda)+'.csv', help="the output file path")

args = parser.parse_args()

# run_prenet_unseenanomaly(args)
run_prenet(args)
