
import matplotlib
matplotlib.use('Agg')
import numpy as np
import time
import os
import pickle
import glob
from compute_salience import UncertaintySalienceAnalyser
from BBalpha_dropout import *
import tensorflow as tf
from keras import backend as K
import utils_classifiers as utlC
import utils_sampling as utlS
import utils_visualise as utlV

dataset = 'covid'
netname = 'bbalpha-run1'
suffix = None
path_to_model = '../models/bbalpha/keras/saved_models/' +\
        '{}-cnn-alpha0.5-run1/model-test.h5'.format(dataset)
test_indices = [2,92]
win_size = 8               
overlapping = True
sampl_style = 'conditional'
num_samples = 10
padding_size = 2            
v_train = np.transpose(x_train, axes=(0, 3, 1, 2))
mode = 'BNN'
net = utlC.BBalpha_keras_net(path_to_model)


def target_func(x):
 
    assert np.ndim(x) == 4, 'Expected x to have 4 dim shape (n, w, h, channels)'
    assert x.shape[-1] == 1 or x.shape[-1] == 3, 'Expected 1 or 3 channels'
    return net.forward_pass(x)


# ------------------------ SET-UP ------------------------

labels =['COVID-19', 'viral pneumonia','bacterial pneumonia','Normal']
results_dir = '/content/drive/My Drive/covid/resultsbntbcv{}-{}/'.format(dataset, netname)
os.makedirs(results_dir, exist_ok=True)


assert mode == 'DNN' or mode == 'BNN', \
        "invalid mode, use 'DNN' or 'BNN' only"

for test_idx in test_indices:

    x_testx = x_test[test_idx]
    x_test_im = x_test[test_idx]

    if mode == 'BNN':
        y_mc = target_func(np.expand_dims(x_testx, axis=0))
        y_avg = np.mean(y_mc, axis=1)
        y_pred = np.argmax(y_avg)
    elif mode == 'DNN':
        y_softmax = target_func(np.expand_dims(x_testx, axis=0))
        y_pred = np.argmax(y_softmax)

    y_true = np.argmax(y_test[test_idx])
    print("Test Image: {}\tPredicted: {}\tTrue:{}".format(test_idx, y_pred, np.argmax(y_test[test_idx])))

    if sampl_style == 'conditional':
        if mode == 'BNN':
            filename = '{}_test{}_winSize{}_condSampl_numSampl{}_padSize{}'.format(dataset, test_idx, win_size, num_samples, padding_size)
        elif mode == 'DNN':
            filename = '{}_DNN_test{}_winSize{}_condSampl_numSampl{}_padSize{}'.format(dataset, test_idx, win_size, num_samples, padding_size)
        path_base = os.path.join(results_dir, filename)
    elif sampl_style == 'marginal':
        filename = '{}_test{}_winSize{}_margSampl_numSampl{}'.format(dataset, test_idx, win_size, num_samples)
        path_base = os.path.join(results_dir, filename)

    save_path = path_base + '_' + suffix if suffix else path_base
    print(save_path)

    existing_results = glob.glob(path_base + '*.p')
    if os.path.exists(save_path+'.p'):
        print('Results for test{} exist, will move to the next'
              ' image.'.format(test_idx))
    elif existing_results:
        print('Results for test{} exist under a different suffix'
              .format(test_idx))
        print('Linking {} to {}'.format(existing_results[0], save_path+'.p'))
        os.system('ln -s {} {}'.format(existing_results[0], save_path+'.p'))
    else:
        print("Analysing...test image: {}, net: {}, window size: {}, sampling: {}"
              .format(test_idx, netname, win_size, sampl_style))

        start_time = time.time()
        
        if sampl_style == 'conditional':
         # if G==2:
          #tf.compat.v1.disable_eager_execution()
	       # print("[INFO] training with {} GPUs...".format(G))

             sampler = utlS.cond_sampler(win_size=win_size,
                                        padding_size=padding_size,
                                        X=v_train, directory=results_dir)
            # sampler = multi_gpu_model(sampler, gpus = 2)
        elif sampl_style == 'marginal':
            sampler = utlS.marg_sampler(win_size=win_size, X=train[0])
        c = np.transpose(x_testx, axes=(2,0,1))
         
        analyser = UncertaintySalienceAnalyser(c, target_func, sampler,
                                               num_samples=num_samples,
                                               mode=mode)
       # if G==2:
          #tf.compat.v1.disable_eager_execution()
	        #  print("[INFO] training with {} GPUs...".format(G))

          #  analyser = multi_gpu_model(analyser, gpus = 2)
        
        salience_dict, counts = analyser.get_rel_vect(win_size=win_size,
                                                      overlap=overlapping)

        salience_dict['y_pred'] = y_pred
        salience_dict['pred_outputs'] = analyser.true_tar_val

        with open(save_path + '.p', 'wb') as f:
            pickle.dump(salience_dict, f)

        print("--- Total computation took {:.4f} seconds  ---".format((time.time() -
                                                                      start_time)))
    # plot and save the results
    existing_results = glob.glob(path_base + '*.png')
    if os.path.exists(save_path+'.png'):
        print('Relevance map for test{} exist, will move to the next'
              ' image.'.format(test_idx))
    elif existing_results:
        print('Results for test{} exist under a different suffix'
              .format(test_idx))
        print('Linking {} to {}'.format(existing_results[0], save_path+'.png'))
        os.system('ln -s {} {}'.format(existing_results[0], save_path+'.png'))
    else:
        print("Plotting...test image: {}, net: {}, window size: {}, sampling: {}"
              .format(test_idx, netname, win_size, sampl_style))
        salience_dict = pickle.load(open(save_path + '.p', 'rb'))
        if mode == 'BNN':
            titles = ['epistemic', 'aleatoric', 'predictive']
            diffs = [salience_dict[x] for x in titles]
            titles.append('pred')
            diffs.append(salience_dict['pred'][:, salience_dict['y_pred']])
        elif mode == 'DNN':
            titles = labels
            diffs = [salience_dict['pred'][:, ii] for ii in range(len(labels))]
        d = np.transpose(x_test_im, axes=(2,0,1))
        utlV.plot_results(d, y_true, salience_dict['y_pred'],
                          diffs, titles, labels,
                         save_path + '.png' )

