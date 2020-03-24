
import pandas as pd
import numpy as np
import random
import os

from neural_network_wrapper import NeuralNetworkWrapper
from data_preprocessing import StandardScaler, one_hot_encode
import optimizers

import matplotlib.pyplot as plt


def prepare_data_regression(train_data, test_data):
    X_train = np.array(train_data.loc[:, ['x']])
    y_train = train_data.y
    y_train = np.array(y_train)

    X_test = np.array(test_data.loc[:, ['x']])
    y_test = test_data.y
    y_test = np.array(y_test)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}


def perform_experiment(dataset,
                       d,
                       exp_objective,
                       exp_values,
                       num_reps):
    """
    """
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    d = d.copy()

    for k in exp_values.keys():
        d[k] = {}
        d[k]['test_rmse'] = []

    for i in range(num_reps):

        for k, v in exp_values.items():
            if exp_objective == 'lr':

                NN = NeuralNetworkWrapper(d['input_dim'],
                                          d['neuron_numbers'],
                                          ['relu'] * (len(d['neuron_numbers']) - 1) + d['output_activation'],
                                          d['loss_function'],
                                          v,
                                          optimizers.Optimizer(),
                                          d['batch_size'],
                                          seed=(d['seed'] + i))

            elif exp_objective == 'activation_function':

                NN = NeuralNetworkWrapper(d['input_dim'],
                                          d['neuron_numbers'],
                                          v * (len(d['neuron_numbers']) - 1) + d['output_activation'],
                                          d['loss_function'],
                                          d['learning_rate'],
                                          optimizers.Optimizer(),
                                          d['batch_size'],
                                          seed=(d['seed'] + i))

            elif exp_objective == 'inertia':

                NN = NeuralNetworkWrapper(d['input_dim'],
                                          d['neuron_numbers'],
                                          ['relu'] * (len(d['neuron_numbers']) - 1) + d['output_activation'],
                                          d['loss_function'],
                                          d['learning_rate'],
                                          optimizers.GDwithMomentum(v),
                                          d['batch_size'],
                                          seed=(d['seed'] + i))

            elif exp_objective == 'batch_size':

                NN = NeuralNetworkWrapper(d['input_dim'],
                                          d['neuron_numbers'],
                                          ['relu'] * (len(d['neuron_numbers']) - 1) + d['output_activation'],
                                          d['loss_function'],
                                          d['learning_rate'],
                                          optimizers.Optimizer(),
                                          v,
                                          seed=(d['seed'] + i))

            NN.train(X_train,
                     y_train,
                     d['num_epochs'],
                     validation_split=0,
                     test_rmse=(X_test, y_test),
                     verbosity=False)

            d[k]['test_rmse'].append(NN.test_rmse)

    for k in exp_values.keys():
        # aggregating results
        d[k]['test_rmse_mean'] = np.mean(np.array(d[k]['test_rmse']).T, axis=1)
        d[k]['test_rmse_std'] = np.std(np.array(d[k]['test_rmse']).T, axis=1)

        d[k] = {"RMSE": d[k]['test_rmse_mean'],
                "RMSE std": d[k]['test_rmse_std'],
                "Best RMSE": np.min(d[k]['test_rmse_mean']),
                "Best RMSE std": d[k]['test_rmse_std'][np.argmin(d[k]['test_rmse_mean'])]}

    return {k: d[k] for k in exp_values.keys()}


def experiments_pipeline(data,
                         experiment_dict,
                         experiments,
                         num_reps=1,
                         save_to_file=False):
    """
    """
    d = experiment_dict.copy()
    output = {'lr': {},
              #'activation_function': {},
              'inertia': {},
              'batch_size': {}}
    # Experiments for each dataset
    for dataset in data:
        print("------ Dataset name: {}".format(dataset['dataset name']))
        output['lr'][dataset['dataset name']] = perform_experiment(dataset['data'],
                                                                   experiment_dict,
                                                                   'lr',
                                                                   experiments['lr'],
                                                                   num_reps)
        # output['activation_function'][dataset['dataset name']] = perform_experiment(dataset['data'],
        #                                                                             experiment_dict,
        #                                                                             'activation_function',
        #                                                                             experiments['activation_function'],
        #                                                                             num_reps)

        output['inertia'][dataset['dataset name']] = perform_experiment(dataset['data'],
                                                                        experiment_dict,
                                                                        'inertia',
                                                                        experiments['inertia'],
                                                                        num_reps)

        output['batch_size'][dataset['dataset name']] = perform_experiment(dataset['data'],
                                                                           experiment_dict,
                                                                           'batch_size',
                                                                           experiments['batch_size'],
                                                                           num_reps)

    return output




# sns.set_style("white")
# sns.despine(left=True, bottom=True)


# Available matplotlib styles:
# ['seaborn-colorblind', 'fast', 'seaborn-deep', 'fivethirtyeight', 'seaborn',
#  'Solarize_Light2', 'seaborn-talk', 'seaborn-darkgrid', 'ggplot', 'seaborn-bright',
#  'seaborn-paper', 'seaborn-pastel', 'bmh', 'seaborn-dark', 'seaborn-notebook',
#  'tableau-colorblind10', 'seaborn-white', '_classic_test', 'seaborn-ticks',
#  'seaborn-poster', 'seaborn-muted', 'dark_background', 'grayscale',
#  'seaborn-dark-palette', 'seaborn-whitegrid', 'classic']

def visualize_experiment(d, title="", figsize=(21, 12)):
    plt.style.use('_classic_test')
    plt.style.use('seaborn-white')
    fig, a = plt.subplots(2, 2)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    dataset_names = list(d.keys())

    fig.suptitle(title, fontsize=30)

    counter = 0
    for i in range(2):
        for j in range(2):
            subplot_dict = d[dataset_names[counter]]

            a[i][j].title.set_text(f'RMSE in {dataset_names[counter]}')
            #a[i][j].set_ylim((0, 1))
            a[i][j].set_xticks(range(1, 1 + len(subplot_dict[list(subplot_dict.keys())[0]]['RMSE'])))

            a[i][j].set_xlabel('Epoch')
            a[i][j].set_ylabel('RMSE')

            for k, v in subplot_dict.items():
                a[i][j].errorbar([it + 1 for it in range(len(v['RMSE']))],
                                 v['RMSE'],
                                 yerr=v['RMSE std'],
                                 linestyle='--',
                                 marker='o',
                                 label=k)

                a[i][j].legend(loc="lower left")

            counter += 1

    # Experiment report
    dataset_names = []
    x = {}
    for dataset_name, exp_results in d.items():
        dataset_names.append(dataset_name)

        for k, v in exp_results.items():
            try:
                x[k].append("{:.2f} +- {:.2f}".format(v['Best RMSE'], v['Best RMSE std']))
            except:
                x[k] = ["{:.2f} +- {:.2f}".format(v['Best RMSE'], v['Best RMSE std'])]

    y = pd.DataFrame(x)
    y.index = dataset_names
    y.transpose()

    return y, fig  # returns dataframe with results & figure


import datetime


def experiment_save_results(d, architecture, path, ds_name, figsize=(21, 12)):

    titles = {'lr': 'Learning rate comparison for {} architecture'.format(str(architecture)),
              'activation_function': 'Activation functions comparison for {} architecture'.format(
                  str(architecture)),
              'inertia': 'Momentum impact comparison for {} architecture'.format(str(architecture)),
              'batch_size': 'Batch size impact comparison for {} architecture'.format(
                  str(architecture))}


    for exp_name, exp_results in d.items():
        output_table, fig = visualize_experiment(exp_results,
                                                 titles[exp_name])
        with open(os.path.join(path,
                               exp_name + "_" + ds_name + "_" + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "_" + ".txt"),
                  "w") as text_file:
            text_file.write(output_table.to_latex())

        fig.savefig(os.path.join(path,
                                 exp_name + "_" + ds_name + "_"+ datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"))


def main():

    data_activation_train_100 = pd.read_csv("./projekt1/regression/data.activation.train.100.csv")
    data_activation_train_500 = pd.read_csv("./projekt1/regression/data.activation.train.500.csv")
    data_activation_train_1000 = pd.read_csv("./projekt1/regression/data.activation.train.1000.csv")
    data_activation_train_10000 = pd.read_csv("./projekt1/regression/data.activation.train.10000.csv")

    data_activation_test_100 = pd.read_csv("./projekt1/regression/data.activation.test.100.csv")
    data_activation_test_500 = pd.read_csv("./projekt1/regression/data.activation.test.500.csv")
    data_activation_test_1000 = pd.read_csv("./projekt1/regression/data.activation.test.1000.csv")
    data_activation_test_10000 = pd.read_csv("./projekt1/regression/data.activation.test.10000.csv")

    data_cube_train_100 = pd.read_csv("./projekt1/regression/data.cube.train.100.csv")
    data_cube_train_500 = pd.read_csv("./projekt1/regression/data.cube.train.500.csv")
    data_cube_train_1000 = pd.read_csv("./projekt1/regression/data.cube.train.1000.csv")
    data_cube_train_10000 = pd.read_csv("./projekt1/regression/data.cube.train.10000.csv")

    data_cube_test_100 = pd.read_csv("./projekt1/regression/data.cube.test.100.csv")
    data_cube_test_500 = pd.read_csv("./projekt1/regression/data.cube.test.500.csv")
    data_cube_test_1000 = pd.read_csv("./projekt1/regression/data.cube.test.1000.csv")
    data_cube_test_10000 = pd.read_csv("./projekt1/regression/data.cube.test.10000.csv")


    data_activation = [{"dataset name": "Data activation 100 obs",
                     "data": prepare_data_regression(data_activation_train_100, data_activation_test_100)},
                    {"dataset name": "Data activation 500 obs",
                     "data": prepare_data_regression(data_activation_train_500, data_activation_test_500)},
                    {"dataset name": "Data activation 1000 obs",
                     "data": prepare_data_regression(data_activation_train_1000, data_activation_test_1000)},
                    {"dataset name": "Data activation 10000 obs",
                     "data": prepare_data_regression(data_activation_train_10000, data_activation_test_10000)}]

    data_cube = [{"dataset name": "Data cube 100 obs",
                     "data": prepare_data_regression(data_cube_train_100, data_cube_test_100)},
                    {"dataset name": "Data cube 500 obs",
                     "data": prepare_data_regression(data_cube_train_500, data_cube_test_500)},
                    {"dataset name": "Data cube 1000 obs",
                     "data": prepare_data_regression(data_cube_train_1000, data_cube_test_1000)},
                    {"dataset name": "Data cube 10000 obs",
                     "data": prepare_data_regression(data_cube_train_10000, data_cube_test_10000)}]



    # CETERIS PARIBUS NETWORK ARCHITECTURE

    arch1 = {
        "input_dim": 1,
        "neuron_numbers": [1],  # number of neurons in consecutive layers
        "activation_functions": [],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }

    arch2 = {
        "input_dim": 1,
        "neuron_numbers": [5, 1],  # number of neurons in consecutive layers
        "activation_functions": ['tanh'],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }
    arch3 = {
        "input_dim": 1,
        "neuron_numbers": [5, 5, 1],  # number of neurons in consecutive layers
        "activation_functions": ['tanh', 'linear'],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }

    arch4 = {
        "input_dim": 1,
        "neuron_numbers": [5, 5, 1],  # number of neurons in consecutive layers
        "activation_functions": ['tanh', 'tanh'],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }

    arch5 = {
        "input_dim": 1,
        "neuron_numbers": [5, 5, 5, 5, 1],  # number of neurons in consecutive layers
        "activation_functions": ['tanh', 'tanh', 'tanh', 'tanh'],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }

    arch6 = {
        "input_dim": 1,
        "neuron_numbers": [5, 5, 5, 5, 1],  # number of neurons in consecutive layers
        "activation_functions": ['relu', 'tanh', 'tanh', 'linear'],
        "loss_function": 'mean_squared_error',
        "batch_size": 64,
        "num_epochs": 20,
        "seed": 42,
        "output_activation": ['linear'],
        "learning_rate": 0.001
    }

    ####

    experiments = {'lr':
                       {'lr=0.00001': 0.00001,
                        'lr=0.0001': 0.0001,
                        'lr=0.001': 0.001,
                        'lr=0.01': 0.01},
                   # 'activation_function':
                   #     {'relu': ['relu'],
                   #      'leaky relu': ['leaky_relu'],
                   #      'sigmoid': ['sigmoid'],
                   #      'tanh': ['tanh']},
                   'inertia':
                       {'beta=0': 0,
                        'beta=0.5': 0.5,
                        'beta=0.9': 0.9},
                   'batch_size':
                       {'bs=4': 4,
                        'bs=16': 16,
                        'bs=32': 32,
                        'bs=64': 64}
                   }

    NUM_REPS=30
    ###
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch1, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch1['neuron_numbers'],
                            path="./report/results/regression/architecture-1",
                            ds_name='data_activation')
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch2, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch2['neuron_numbers'],
                            path="./report/results/regression/architecture-2",
                            ds_name='data_activation')
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch3, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch3['neuron_numbers'],
                            path="./report/results/regression/architecture-3",
                            ds_name='data_activation')
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch4, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch4['neuron_numbers'],
                            path="./report/results/regression/architecture-4",
                            ds_name='data_activation')
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch5, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch5['neuron_numbers'],
                            path="./report/results/regression/architecture-5",
                            ds_name='data_activation')
    print("=" * 40)
    ans = experiments_pipeline(data_activation, arch6, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch6['neuron_numbers'],
                            path="./report/results/regression/architecture-6",
                            ds_name='data_activation')




    ans = experiments_pipeline(data_cube, arch1, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch1['neuron_numbers'],
                            path="./report/results/regression/architecture-1",
                            ds_name='data_cube')

    ans = experiments_pipeline(data_cube, arch2, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch2['neuron_numbers'],
                            path="./report/results/regression/architecture-2",
                            ds_name='data_cube')

    ans = experiments_pipeline(data_cube, arch3, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch3['neuron_numbers'],
                            path="./report/results/regression/architecture-3",
                            ds_name='data_cube')

    ans = experiments_pipeline(data_cube, arch4, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch4['neuron_numbers'],
                            path="./report/results/regression/architecture-4",
                            ds_name='data_cube')

    ans = experiments_pipeline(data_cube, arch5, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch5['neuron_numbers'],
                            path="./report/results/regression/architecture-5",
                            ds_name='data_cube')

    ans = experiments_pipeline(data_cube, arch6, experiments, num_reps=NUM_REPS)
    experiment_save_results(ans, arch6['neuron_numbers'],
                            path="./report/results/regression/architecture-6",
                            ds_name='data_cube')



if __name__ == "__main__":
    main()