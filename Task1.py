import andi 
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses, metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
import os
AD = andi.andi_datasets()


#Building the network
model_norm_long = Sequential()
#first layer: LSTM of dimension 64
model_norm_long.add(LSTM(250,
                return_sequences=True,
                recurrent_dropout=0.2,
                input_shape=(None, 4)
                ))

model_norm_long.add(LSTM(50,
                    dropout=0,
                    recurrent_dropout=0.2,
                    ))

#Last layer, fully connected
model_norm_long.add(Dense(1))
model_norm_long.compile(optimizer='adam',
                loss='mse', 
                metrics=['mae'])

#Printing a summary of the built network
model_norm_long.summary()


i = 125
N = 100000
dimension = 1
X1, Y1, Z, Z, Z, Z = AD.andi_dataset(N = N, tasks = 1, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=True, path_datasets=str(i)+'master1')

X2, Y2, Z, Z, Z, Z = AD.andi_dataset(N = N, tasks = 1, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=True, path_datasets=str(i)+'master2')

X3, Y3, Z, Z, Z, Z = AD.andi_dataset(N = N, tasks = 1, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=True, path_datasets=str(i)+'master3')

X1[0] = np.diff(X1[0],axis=1)
X2[0] = np.diff(X2[0],axis=1)
X3[0] = np.diff(X3[0],axis=1)

data_tot=X1[0]
data_norm1 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))
data_tot=X2[0]
data_norm2 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))
data_tot=X3[0]
data_norm3 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))


Y1 = np.array(Y1[0])
Y2 = np.array(Y2[0])
Y3 = np.array(Y3[0])




for n in range(1):
    
    for batch_size in [32, 128, 512, 2048,]:
        
        for repeat in range(20):
            traj_show = data_norm1
            label_show = Y1
            history_norm_long = model_norm_long.fit(traj_show.reshape(len(traj_show),int(i/4),4), 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' +  str(5*repeat + 3) +'percent')
            traj_show = data_norm2
            label_show = Y2
            history_norm_long = model_norm_long.fit(traj_show.reshape(len(traj_show),int(i/4),4), 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' + str(5*repeat + 7) +'percent')
            traj_show = data_norm3
            label_show = Y3
            history_norm_long = model_norm_long.fit(traj_show.reshape(len(traj_show),int(i/4),4), 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' + str(5*repeat + 10) +'percent')
            model_norm_long.save('checkpoint.h5')
           
        
model_norm_long.save('Model_1D_recdout_' + str(i) + '.h5')







