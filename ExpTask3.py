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

#import os
#AD = andi.andi_datasets()




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
model_switch.add(Dense(10))
model_switch.compile(optimizer='adam',
                loss='mse', 
                metrics=['mae'])

#Printing a summary of the built network
#model_switch.summary()


x=np.load('switch_tr.npy')
Y1=np.load('switch_label.npy')


#label for changing point and alpha. entries from labels are (t, a1, m1, l1, j1, a2, m2, l2,j2) 
#I want (a1,m1,l1,j1, a2, m2, l2, j2, sin(2pi*tc/T), cos(2pi*tc/T))


ccv=Y1
ss=np.shape(ccv)
ccl=np.zeros((ss[0],10))
for kk in range(8):
    ccl[:,kk]=ccv[:,1+kk]

ccl[:,8]=np.sin((2*np.pi*ccv[:,0])/200)
ccl[:,9]=np.cos((2*np.pi*ccv[:,0])/200)


i=200
test_tim_step=np.arange(i)
show_time_coll=np.tile(test_tim_step,(len(x),1))

mc = keras.callbacks.ModelCheckpoint(
    filepath='exp_checkpoint_1D_recdout_' + str(i) + '.h5',
    save_weights_only=False,
    monitor='val_mae',
    mode='auto',
    save_best_only=True)

for n in range(1):
    
    for batch_size in [32, 128, 256]:
        
        for repeat in range(5):
            #Normalize trajectory, makes it into array of dimension 2, using bot position and time stamp
            data_show,label_show,traj_show,times_show=data_split(np.asarray(x),
                                                     show_time_coll,
                                                         labels=ccl,
                                                         start_row=0,num_row=len(x),
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
                  
            model_switch.save('exp_checkpoint.h5')
           
        
model_switch.save('Exp_Task3_Model_1D_recdout_' + str(i) + '.h5')
