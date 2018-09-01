import numpy as np

from nn.alexnet import alexnet
from utils.constants import *
from utils.file.file_merge import merge_npy_files

WIDTH = 160
HEIGHT = 120
LR = 1e-3
N_EPOCH = 8

MODEL_NAME = "PythonGTA5_SUPERBIKE_{}_{}_{}_Epochs.model".format(LR, "alexnet", N_EPOCH)
model = alexnet(WIDTH, HEIGHT, LR)

data = merge_npy_files(BIKE_TRAINING_DATA_PATH)

train = data[:-200]
test = data[-200:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

# model.fit({'input': X}, {'target': Y}, n_epoch=N_EPOCH, validation_set=(test_x, test_y),
#           snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.fit({'input': X}, {'targets': Y}, n_epoch=N_EPOCH, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_DIR + "/" + MODEL_NAME)
