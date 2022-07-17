import os
import logging
import numpy as np
import matplotlib.pyplot as plt

def create_logger(save_path='', file_type='', level='debug', console=True):
    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    if console:
        cs = logging.StreamHandler()
        cs.setLevel(_level)
        logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def plot_single_curve(path, name, point, freq=1, xlabel='Epoch',ylabel=None):
    
    x = (np.arange(len(point)) + 1) * freq
    plt.plot(x, point, color='purple')
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = name
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()

def plot_curves(path, name, point_list, curve_names=None, freq=1, xlabel='Epoch',ylabel=None):
    if curve_names is None:
        curve_names = [''] * len(point_list)
    else:
        assert len(point_list) == len(curve_names)

    x = (np.arange(len(point_list[0])) + 1) * freq
    if len(point_list) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')
    for i, (point, curve_name) in enumerate(zip(point_list, curve_names)):
        assert len(point) == len(x)
        plt.plot(x, point, color=cmap(i), label=curve_name)
        
    plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()