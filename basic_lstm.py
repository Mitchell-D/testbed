
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

nldas1D = Sequential()

nldas1D.add(InputLayer((5, 1)))
nldas1D.add(LSTM(64))
nldas1D.add(Dense(8, 'relu'))
nldas1D.add(Dense(1, 'linear'))

nldas1D.summary()
