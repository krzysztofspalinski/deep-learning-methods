import matplotlib.pyplot as plt
import networkx as nx
import os
from PIL import Image


def make_network_state_gif(cached_network_states,
                           imgs_dir,
                        file_name):

        def _draw_network_states(cached_network_states, imgs_dir):
            for epoch in range(len(cached_network_states)):

                cache = cached_network_states[epoch]
                architecture = [layer.shape[0] for layer in cache]
                architecture = [cache[0].shape[1]] + architecture

                counter = 0
                pos = {}
                for i in range(len(architecture)):
                    for j in range(architecture[i]):
                        pos[f"W{i}_{j}"] = (i, j)
                        counter += 1

                G = nx.Graph()
                for k, v in pos.items():
                    G.add_node(k, pos=v)

                weights = [layer for layer in cache]
                for w in weights: w *= 10

                for num_layer in range(len(weights)):
                    num_row, num_col = weights[num_layer].shape
                    for i in range(num_row):
                        for j in range(num_col):
                            G.add_edge(f"W{num_layer}_{j}", f"W{num_layer + 1}_{i}", weight=weights[num_layer][i, j])

                edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
                nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)

                plt.savefig(f"{imgs_dir}_state_{epoch+1}_{len(cached_network_states)}.png")
                print(f"State {epoch + 1}/{len(cached_network_states)}")
            return

        _draw_network_states(cached_network_states, imgs_dir)

        files = os.listdir(imgs_dir)
        pic_names = []
        for file in files:
            if '.png' in file:
                pic_names.append(file)
        pic_names = sorted(pic_names)

        frames = []

        for picture in pic_names:
            new_frame = Image.open(imgs_dir + picture)

            for i in range(10):
                frames.append(new_frame)

        for _ in range(30):
            frames.append(frames[-1])

        if file_name is None:
            file_name = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

        if not os.path.isfile(file_name):
            frames[0].save(file_name, format='GIF',
                           append_images=frames[1:],
                           save_all=True,
                           duration=175, loop=0)
        else:
            raise Exception(f"File {file_name} already exists!")

        for frame in frames:
            frame.close()
