from neural_network_core import NeuralNetworkCore
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import os
import numpy as np
import shutil
from data_preprocessing import one_hot_encode


class LearningProcessVisualisation:
    """
    Creates animation of learning process for 2d datasets.
    y_test/train should consist of m x 1 column vector of numbers 1, 2, ..., #classes.
    """

    def __init__(self, layers, activation_functions, learning_rate):
        self.layers = layers
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate

    def animate_2D(self, X_train, y_train, X_test, y_test,
                   iterations,
                   file_name=None,
                   animation_step=5,
                   save_path="./animation/"
                   ):

        y_train = one_hot_encode(y_train)

        NeuralNet = NeuralNetworkCore(X_train.shape[1],
                                      self.layers,
                                      self.activation_functions,
                                      "max_likelihood_loss",
                                      self.learning_rate)

        tmp_path = "./.TMP/"
        self.__create_tmp(tmp_path)

        n_of_grid_points = 50

        x_min, x_max, y_min, y_max = self.__get_min_max_xy(X_train)
        shift_x = abs(x_max - x_min) / 8
        shift_y = abs(y_max - y_min) / 8
        x_grid, y_grid = np.meshgrid(np.linspace(x_min - shift_x, x_max + shift_x, n_of_grid_points),
                                     np.linspace(y_min - shift_y, y_max + shift_y, n_of_grid_points))
        grid_predict = np.column_stack((np.reshape(x_grid, (-1, 1)), np.reshape(y_grid, (-1, 1))))

        for i in range(iterations):
            loss = NeuralNet.train(X_train, y_train, 1)

            if (i + 1) % animation_step == 0:
                grid_predict_hat = NeuralNet.predict(grid_predict)
                grid_predict_hat = grid_predict_hat.argmax(1)

                xy_grid_hat = np.reshape(grid_predict_hat, (n_of_grid_points, n_of_grid_points))

                plt.figure(figsize=(9, 6))
                plt.axis('off')
                plt.contourf(x_grid, y_grid, xy_grid_hat)
                plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, animated=True, edgecolors='black')
                textbox_message = f'Iteration: {i + 1} \n Loss: {loss[0]:.4f}'
                plt.text(x_min - shift_x / 2, y_max + shift_y / 2,
                         textbox_message,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                plt.savefig(f'{tmp_path}iter_{i + 1}.png', dpi=70, bbox_inches='tight')
                plt.close('all')

                print(loss)

        self.__make_gif(tmp_path, save_path, file_name)
        self.__delete_tmp(tmp_path)

    def animate_1D(self, X_train, y_train, X_test, y_test,
                   iterations,
                   file_name=None,
                   animation_step=5,
                   save_path="./animation/"
                   ):

        NeuralNet = NeuralNetworkCore(X_train.shape[1],
                                      self.layers,
                                      self.activation_functions,
                                      "root_mean_squared_error",
                                      self.learning_rate)

        tmp_path = "./.TMP/"
        self.__create_tmp(tmp_path)

        n_of_grid_points = 100

        x_min, x_max = np.min(X_train), np.max(X_train)
        y_min, y_max = np.min(y_train), np.max(y_train)

        shift_x = abs(x_max - x_min) / 8
        shift_y = abs(y_max - y_min) / 8

        x_grid = np.linspace(x_min - shift_x, x_max + shift_x, n_of_grid_points)
        x_grid = np.reshape(x_grid, (n_of_grid_points, 1))

        for i in range(iterations):
            loss = NeuralNet.train(X_train, y_train, 1)

            if (i + 1) % animation_step == 0:
                y_grid_hat = NeuralNet.predict(x_grid)

                plt.figure(figsize=(9, 6))

                plt.axis((x_min - shift_x, x_max + shift_x, y_min - shift_y, y_max + shift_y))
                plt.scatter(X_test, y_test, animated=True, edgecolors='black')
                plt.plot(x_grid, y_grid_hat, animated=True, color='red')
                textbox_message = f'Iteration: {i + 1} \n Loss: {loss[0]:.4f}'
                plt.text(x_min - shift_x / 2, y_max + shift_y / 2,
                         textbox_message,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                plt.savefig(f'{tmp_path}iter_{i + 1}.png', dpi=70, bbox_inches='tight')
                plt.close('all')

                print(loss)

        self.__make_gif(tmp_path, save_path, file_name)
        self.__delete_tmp(tmp_path)

    @staticmethod
    def __create_tmp(tmp_path):
        try:
            shutil.rmtree(tmp_path)
        except FileNotFoundError:
            pass
        os.mkdir(tmp_path)

    @staticmethod
    def __delete_tmp(tmp_path):
        try:
            shutil.rmtree(tmp_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def __make_gif(pictures_path, save_path, file_name):
        files = os.listdir(pictures_path)
        pic_names = []
        for file in files:
            if '.png' in file:
                pic_names.append(file)
        pic_names = sorted(pic_names, key=lambda x: int(x[5:-4]))

        frames = []

        for picture in pic_names:
            new_frame = Image.open(pictures_path + picture)
            frames.append(new_frame)

        for _ in range(30):
            frames.append(frames[-1])

        if file_name is None:
            file_name = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass

        if os.path.isfile(f'{save_path}{file_name}.gif'):
            date_and_time = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            frames[0].save(f'{save_path}{file_name}_{date_and_time}.gif', format='GIF',
                           append_images=frames[1:],
                           save_all=True,
                           duration=175, loop=0)
        else:
            frames[0].save(f'{save_path}{file_name}.gif', format='GIF',
                           append_images=frames[1:],
                           save_all=True,
                           duration=175, loop=0)
        for frame in frames:
            frame.close()

    @staticmethod
    def __get_min_max_xy(X):
        x_min = np.min(X[:, 0])
        x_max = np.max(X[:, 0])
        y_min = np.min(X[:, 1])
        y_max = np.max(X[:, 1])
        return x_min, x_max, y_min, y_max
