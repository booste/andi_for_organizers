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
from data_split import data_split

import os
AD = andi.andi_datasets()


#Building the network
model_switch = Sequential()
#first layer: LSTM of dimension 64
model_switch.add(LSTM(250,
                return_sequences=True,
                recurrent_dropout=0.2,
                input_shape=(None, 2)
                ))

model_switch.add(LSTM(50,
                    dropout=0,
                    recurrent_dropout=0.2,
                    ))

#Last layer, fully connected
model_switch.add(Dense(4))
model_switch.compile(optimizer='adam',
                loss='mse', 
                metrics=['mae'])

#Printing a summary of the built network
model_switch.summary()


i = 200
N = 10
dimension = 1
Z, Z, Z, Z, X1, Y1 = AD.andi_dataset(N = N, tasks = 3, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=False, path_datasets=str(i)+'master1')

Z, Z, Z, Z, X2, Y2 = AD.andi_dataset(N = N, tasks = 3, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=False, path_datasets=str(i)+'master2')

Z, Z, Z, Z, X3, Y3 = AD.andi_dataset(N = N, tasks = 3, dimensions = dimension,
                                             min_T = i, max_T = i+1, load_dataset=False, path_datasets=str(i)+'master3')
print('If it is not the first time you use this, load the files instead of creating them!',
      'If you load them change the parameter corr from 0 to 1')
#X1[0] = np.diff(X1[0],axis=1)
#X2[0] = np.diff(X2[0],axis=1)
#X3[0] = np.diff(X3[0],axis=1)




Y1 = np.array(Y1[0])
Y2 = np.array(Y2[0])
Y3 = np.array(Y3[0])
#print(Y1)
#label for changing point and alpha. entries from andi are (dimension, tc,class1,a1,class2,a2) 
#I want (a1,a2,sin(2pi*tc/T),cos(2pi*tc/T))
#NB it seems like the data shape chages when you produce or load the data!!! when you create it it also has the entry telling you the dimension
# if working with loaded data use corr=1 otherwise 0
corr=0

ss=np.shape(Y1)
#print('shape is ',ss)
ccl1=np.zeros((ss[0],4))
ccl1[:,0]=Y1[:,3-corr]
ccl1[:,1]=Y1[:,5-corr]
ccl1[:,2]=np.sin((2*np.pi*Y1[:,1-corr])/200)
ccl1[:,3]=np.cos((2*np.pi*Y1[:,1-corr])/200)

#print(ccl1)

ccl2=np.zeros((ss[0],4))
ccl2[:,0]=Y2[:,3-corr]
ccl2[:,1]=Y2[:,5-corr]
ccl2[:,2]=np.sin((2*np.pi*Y2[:,1-corr])/200)
ccl2[:,3]=np.cos((2*np.pi*Y2[:,1-corr])/200)


ccl3=np.zeros((ss[0],4))
ccl3[:,0]=Y3[:,3-corr]
ccl3[:,1]=Y3[:,5-corr]
ccl3[:,2]=np.sin((2*np.pi*Y3[:,1-corr])/200)
ccl3[:,3]=np.cos((2*np.pi*Y3[:,1-corr])/200)



#data_tot=X1[0]
#data_norm1 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))
#data_tot=X2[0]
#data_norm2 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))
#data_tot=X3[0]
#data_norm3 = np.array(( data_tot - np.mean(data_tot, axis=1).reshape(len(data_tot),1) ) / np.std(data_tot,axis=1).reshape(len(data_tot),1))


test_tim_step=np.arange(i)
show_time_coll=np.tile(test_tim_step,(len(X1[0]),1))

mc = keras.callbacks.ModelCheckpoint(
    filepath='checkpoint_1D_recdout_' + str(i) + '.h5',
    save_weights_only=False,
    monitor='val_mae',
    mode='auto',
    save_best_only=True)

for n in range(1):
    
    for batch_size in [32, 128, 512, 2048,]:
        
        for repeat in range(20):
            #Normalize trajectory, makes it into array of dimension 2, using bot position and time stamp
            data_show,label_show,traj_show,times_show=data_split(np.asarray(X1[0]),
                                                     show_time_coll,
                                                         labels=ccl1,
                                                         start_row=0,num_row=len(X1[0]),
                                                         traj_len=i,n_in=0,n_samples=1,
                                                         p_p=1,hmin=0,hmax=2,limith=False,normalization=True)


            history_norm_long = model_switch.fit(data_show, 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,verbose=2,
                                    callbacks=[mc],
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' +  str(5*repeat + 3) +'percent')
            data_show,label_show,traj_show,times_show=data_split(np.asarray(X2[0]),
                                                     show_time_coll,
                                                         labels=ccl2,
                                                         start_row=0,num_row=len(X2[0]),
                                                         traj_len=i,n_in=0,n_samples=1,
                                                         p_p=1,hmin=0,hmax=2,limith=False,normalization=True)

            history_norm_long = model_switch.fit(data_show, 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,verbose=2,
                                    callbacks=[mc],
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' + str(5*repeat + 7) +'percent')
            data_show,label_show,traj_show,times_show=data_split(np.asarray(X3[0]),
                                                     show_time_coll,
                                                         labels=ccl3,
                                                         start_row=0,num_row=len(X3[0]),
                                                         traj_len=i,n_in=0,n_samples=1,
                                                         p_p=1,hmin=0,hmax=2,limith=False,normalization=True)

            history_norm_long = model_switch.fit(data_show, 
                                    label_show, 
                                    epochs=1, 
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    shuffle=True,verbose=2,
                                    callbacks=[mc],
                                    )
            print('########## TRAINING PROGRESS: Batch_size_' + str(batch_size) + '_' + str(5*repeat + 10) +'percent')
            model_switch.save('checkpoint.h5')
           
        
model_switch.save('Task3_Model_1D_recdout_' + str(i) + '.h5')







