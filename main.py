from gng3d_reconstruction import GrowingNeuralGas
from image_io.io import load_open3d, save_es_vs_json
from image_io.plot_gng import plot_gng3d

path_single_pcd = "./database/bag2pcdRGB"
path_es_vs_result = "./result"

max_neurons = 2000
max_iter = 500
max_age = 250
eb = 0.1
en = 0.001
alpha = 0.95
beta = 0.5
l = 100
plot_gng = True
train = False

if (train is True):
    _, dataset, _ = load_open3d(path_single_pcd)

    gng = GrowingNeuralGas(max_neurons, max_iter, max_age, eb, en, alpha, beta, l, dataset[0])
    gng3d = gng.learn()
    save_es_vs_json(gng3d, path_es_vs_result)

if (plot_gng is True):
    plot_gng3d(path_es_vs_result)




