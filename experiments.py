import pandas as pd
import numpy as np
import random
import json
from neural_network_wrapper import NeuralNetworkWrapper
import optimizers
import os

def experiments_pipeline(X, y, experiment_dict, save_to_file=False):
    """
    Function automates
    :param X: training data array
    :param y: dependent variable
    :param experiment_dict: experiment setup:
    "input_dim" : number of independent variables
    "neuron_numbers" : number of neurons in consecutive layers
    "activation_functions" : activation functions in consecutive layers
    "loss_function" : loss_function
    "learning_rate" : learning_rate
    "optimizer" : optimizer
    "batch_size" : batch_size
    "validation_split" : vaildation split
    "num_epochs" : number of epochs
    "seed" : seed to provide reproducibility
    "dataset_name" : dataset_name - part of output file name
    "experiment_name" : experiment_name - part of output file name
    :param save_to_file: triggers if output has to be saved in JSON file
    :return: experiment dict with train/val loss on end of each epoch
    """
    d = experiment_dict.copy()

    # reproducibility issues
    random.seed(d['seed'])
    np.random.seed(d['seed'])

    NN = NeuralNetworkWrapper(d['input_dim'],
                              d['neuron_numbers'],
                              d['activation_functions'],
                              d['loss_function'],
                              d['learning_rate'],
                              d['optimizer'],
                              d['batch_size'])
    NN.train(X,
             y,
             d['num_epochs'],
             d['validation_split'])

    d['loss_on_epoch'] = NN.loss_on_epoch
    d['loss_on_epoch_valid'] = NN.loss_on_epoch_valid

    # TODO: how to evaluate optimizers? Object cannot be saved to a JSON file
    try:
        del d['optimizer']
    except:
        pass

    if save_to_file:
        filename = d['experiment_name'] + '_' + d['dataset_name'] + '.json'

        if filename in os.listdir():
            raise Exception(f"File {filename} already exists!")
        else:
            with open(filename, 'w') as file:
                json.dump(d, file)
            print("File successfully saved!")

    return d

def read_experiment(filepath):
    """
    handy way to read experiment data
    :param filepath:
    :return:
    """
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d



def main():
    data = pd.read_csv("./projekt1/classification/data.simple.train.1000.csv")

    X = np.array(data.loc[:, ['x', 'y']])
    y = data.cls
    y -= 1
    # one hot encoding
    y_ohc = np.zeros((y.size, int(np.max(y)) + 1))
    y_ohc[np.arange(y.size), y.astype(np.int)] = 1
    y = y_ohc

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X = ss.fit_transform(X)

    input_dim = 2
    neuron_numbers = [4, 4, 2]
    activation_functions = ['relu', 'relu', 'sigmoid']
    loss_function = 'logistic_loss'
    learning_rate = 0.01
    optimizer = optimizers.Optimizer()
    batch_size = 128
    val_split = 0.1
    num_epochs = 50
    seed = 42
    dataset_name = "test"
    experiment_name = "test1"

    experiment_dict = {
        "input_dim": input_dim,
        "neuron_numbers": neuron_numbers,  # number of neurons in consecutive layers
        "activation_functions": activation_functions,
        "loss_function": loss_function,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "validation_split": val_split,
        "num_epochs": num_epochs,
        "seed": seed,
        "dataset_name": dataset_name,
        "experiment_name": experiment_name
    }

    output = experiments_pipeline(X, y, experiment_dict, True)
    print(read_experiment(experiment_dict['experiment_name'] + '_' + experiment_dict['dataset_name'] + '.json'))

if __name__ == "__main__":
    main()