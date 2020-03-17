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

    def animate(self, X_train, y_train, X_test, y_test,
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
        try:
            shutil.rmtree(tmp_path)
        except FileNotFoundError:
            pass

        os.mkdir(tmp_path)

        x_min, x_max, y_min, y_max = self.__get_min_max_xy(X_train)
        shift_x = abs(x_max - x_min) / 8
        shift_y = abs(y_max - y_min) / 8

        x_grid, y_grid = np.meshgrid(np.linspace(x_min - shift_x, x_max + shift_x, 50),
                             np.linspace(y_min - shift_y, y_max + shift_y, 50))
        grid_predict = np.column_stack((np.reshape(x_grid, (-1, 1)), np.reshape(y_grid, (-1, 1))))

        for i in range(iterations):
            loss = NeuralNet.train(X_train, y_train, 1)

            if (i + 1) % animation_step == 0:
                grid_predict_hat = NeuralNet.predict(grid_predict)
                grid_predict_hat = grid_predict_hat.argmax(1)

                xy_grid_hat = np.reshape(grid_predict_hat, (50, 50))

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
