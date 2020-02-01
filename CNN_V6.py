"""
Copyright (c) 2017-2020 Peking University. All rights
                        reserved.

 * Author of this revised version: Professor Xun Huang, huangxun@pku.edu.cn
"""

"""
The code demonstrates the learning of partial differential d/dx in the presence of solid boundary wall. 
More details can be found in the separate article: 
"Deep neural networks for waves assisted by the Wiener--Hopf method". 
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.io import loadmat
from keras.layers import Dropout
import sys
from pathlib import Path
import numpy as np
import keras_cifar10_trained_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, BatchNormalization
from keras.layers.merge import concatenate, add
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 1. Load trainging data.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_name="xun_traindxdr_nn3_300"
result_dict=loadmat(train_name)
print("type of reuslt:",type(result_dict))
print("keys:",result_dict.keys())

# Load training data
rr1  = np.array(result_dict['rr1'])     
zz1  = np.array(result_dict['zz1'])     
psi1 = np.array(result_dict['Psi_nf1'])
u_nf1 = np.array(result_dict['u_nf1'])
x_num = np.array(result_dict['x_num']) 
y_num = np.array(result_dict['y_num']) 
mm = np.array(result_dict['mm']) 
nn = np.array(result_dict['nn']) 
Amp_in = np.array(result_dict['Amp_in']) 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 2. To use conv network, transfer 1D input to 2D image data
# psi2 = np.zeros([psi1.shape[1],int(y_num[0,0]),int(x_num[0,0])],dtype=np.complex_)
# u_nf2= np.zeros([u_nf1.shape[1],int(y_num[0,0]),int(x_num[0,0])],dtype=np.complex_)
# for single channel input & output
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
psi2 = np.zeros([psi1.shape[0],y_num.item(),x_num.item()],dtype=np.complex_)
u_nf2= np.zeros([u_nf1.shape[0],y_num.item(),x_num.item()],dtype=np.complex_)
# for single channel input & output
psi3 = np.zeros([psi1.shape[0],y_num.item(),x_num.item(),2])
u_nf3= np.zeros([psi1.shape[0],y_num.item(),x_num.item(),2])
id=0
while id < psi1.shape[0]:
    tmp=psi1[id,:]
    tmp=tmp.reshape([y_num.item(),x_num.item()]) 
    psi2[id,:]=tmp
    psi3[id,:,:,0]=tmp.real
    psi3[id,:,:,1]=tmp.imag    
    tmp=u_nf1[id,:]
    tmp=tmp.reshape([y_num.item(),x_num.item()]) 
    u_nf2[id,:]=tmp    
    u_nf3[id,:,:,0]=tmp.real      
    u_nf3[id,:,:,1]=tmp.imag            
    id=id+1
    
del psi1,u_nf1,tmp


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 3. Display the first 5 results to check the input training data
# The data pairs are: psi and u, from which we can train and get d/dx, 
# because according to acoustic theory, particle velocity u = d phi/dx.  
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rr=rr1.reshape([int(y_num.item()),int(x_num.item())]) #zz.shape)
zz=zz1.reshape([int(y_num.item()),int(x_num.item())]) #zz.shape)
id=0
while id < 5:
    bar_range = np.linspace(-1.0,1.0,10,endpoint=True)        
    psi=psi3[id,:,:,0]
    u_nf=u_nf3[id,:,:,0]
    fig2, ax2 = plt.subplots(constrained_layout=True)
    CS=ax2.contourf(zz,rr,u_nf.real/u_nf.real.max(), bar_range, cmap='RdGy')
    ax2.set_xlabel('x', fontsize=15,fontname='Times New Roman')
    ax2.set_ylabel('r', fontsize=15,fontname='Times New Roman')
    ax2.set_title('u-velocity', fontsize=18,fontname='Times New Roman')
    cbar = plt.colorbar(CS)
    plt.savefig('u_'+str(id)+'.jpg')
    plt.clf()    
    fig1, ax1 = plt.subplots(constrained_layout=True)
    CS1=ax1.contourf(zz,rr,psi.real/psi.real.max(), bar_range, cmap='RdGy')
    ax1.set_xlabel('x', fontsize=15,fontname='Times New Roman')
    ax1.set_ylabel('r', fontsize=15,fontname='Times New Roman')
    ax1.set_title('Acoustic potential', fontsize=18,fontname='Times New Roman')
    cbar = plt.colorbar(CS1)
    plt.savefig('psi_'+str(id)+'.jpg')
    plt.clf()
    id=id+1


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 4. Define the deep neural network. 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Conv2D block. Here I disable batchnorm layers.  
def conv2d_block(input_tensor, n_filters, kernel_size=7):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
    x = Activation("relu")(x)          
    # 5th layer
    x = Conv2D(filters=n_filters*2, kernel_size=(kernel_size-2, kernel_size-2), kernel_initializer="he_normal",
               padding="same")(x)
    x = Activation("relu")(x)         
    return x
    
# Define unet3 with more layers and different activation functions
def get_unet3(input_img, n_filters=16, dropout=0.5, kernel_size=7):
    # contracting path
    c1 = conv2d_block_tanh(input_img, n_filters=n_filters*4, kernel_size=kernel_size)    #21
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)
    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=kernel_size)           #22
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=kernel_size-2)         #23
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters=n_filters*4, kernel_size=kernel_size-2)         #24
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    c5 = conv2d_block(p4, n_filters=n_filters*8, kernel_size=kernel_size-4)        #25
    p05 = MaxPooling2D(pool_size=(2, 2)) (c5)
    p05 = Dropout(dropout)(p05)
    c06 = conv2d_block(p05, n_filters=n_filters*8, kernel_size=kernel_size-4)        
    p06 = MaxPooling2D(pool_size=(2, 2)) (c06)
    p06 = Dropout(dropout)(p06)
    c07 = conv2d_block(p06, n_filters=n_filters*8, kernel_size=kernel_size-4)    
    p07 = MaxPooling2D(pool_size=(2, 2)) (c07)
    p07 = Dropout(dropout)(p07)
    c08 = conv2d_block(p07, n_filters=n_filters*16, kernel_size=kernel_size-4)                
    # expansive path
    u08 = Conv2DTranspose(n_filters*8, kernel_size=(kernel_size-4,kernel_size-4), strides=(2, 2), padding='same') (c07)
    u08 = concatenate([u08, c06])
    u08 = Dropout(dropout)(u08)
    c08 = conv2d_block(u08, n_filters=n_filters*8, kernel_size=kernel_size-4)  
    u09 = Conv2DTranspose(n_filters*8, kernel_size=(kernel_size-4,kernel_size-4), strides=(2, 2), padding='same') (c08)
    u09 = concatenate([u09, c5])
    u09 = Dropout(dropout)(u09)
    c09 = conv2d_block(u09, n_filters=n_filters*8, kernel_size=kernel_size-2)      
    u6 = Conv2DTranspose(n_filters*4, kernel_size=(kernel_size-4,kernel_size-4), strides=(2, 2), padding='same') (c09)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=kernel_size-2)         #Transpose 9 
    u7 = Conv2DTranspose(n_filters*4, (kernel_size-2,kernel_size-2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=kernel_size)
    u8 = Conv2DTranspose(n_filters*2, (kernel_size, kernel_size), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=kernel_size)
    u9 = Conv2DTranspose(n_filters*4, (kernel_size, kernel_size), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=kernel_size)    
#    c9 = conv2d_block_linear(u9, n_filters=n_filters*4, kernel_size=kernel_size)      
    c9 = conv2d_block_tanh(u9, n_filters=n_filters*4, kernel_size=kernel_size)    
    outputs = Conv2D(2, (1, 1), activation='linear') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



# The activation layer is replaced with linear function 
def conv2d_block_linear(input_tensor, n_filters, kernel_size=7):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
    x = Activation("linear")(x)                
    return x



# The activation layer is replaced with Tanh function 
def conv2d_block_tanh(input_tensor, n_filters, kernel_size=7):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
    x = Activation("tanh")(x)          
    # 2nd layer  enable of this would lead to blow up of the training. xun, Oct 2019
    x = Conv2D(filters=n_filters*2, kernel_size=(kernel_size-2, kernel_size-2), kernel_initializer="he_normal",
               padding="same")(x)
    x = Activation("tanh")(x)         
    return x



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 5. Start maching learning by building up the NN model 
# Input: psi2 (2D psi field); output: u,v,p fields (2D) 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Build a UNet
input_tensor=Input((int(y_num.item()),int(x_num.item()),2),name='input')
# Note! Disable batchnormalization. Otherwise, training blows up. Xun
# Note: the hyperparameters, n_filter (preferred 8) & kernel_size (preferred 9), are 
#  reduced to smaller values to enable a quick study 
model = get_unet3(input_tensor, n_filters=4, dropout=0.05, kernel_size=7)
model.summary()


model.compile(optimizer='adam', loss='mean_squared_error')
# Define callbacks for model.fit
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)]


# Start the loop of training 
# Users can simply use the following cmd
#
# model_loss=model.fit(psi3, u_nf3, validation_split = 0.2, epochs=50, batch_size=50, callbacks=callbacks)
#
# or the folloing code section to store maching training files during the training. 
#
id0=0
import time
while id0 < 16:
    id0
    model_loss=model.fit(psi3, u_nf3, validation_split = 0.2, epochs=5, batch_size=50, callbacks=callbacks)
    # Save the trained model and weights
    model_name="unetmodel_"+train_name+"id"+str(id0)+".h5"
    weights_name="weights"+train_name+".h5"
    model.save(model_name)
    model.save_weights(weights_name)
    id0=id0+1
    time.sleep(200)     # sleep for 100s to cool down the GPU 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 6. Analysis of the training. 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(model_loss.history.keys())
print(model_loss.history['loss'])
loss=model_loss.history['loss']
val_loss=model_loss.history['val_loss']
plt.plot(loss,lw=2, label='Training loss',color='gray')
plt.plot(val_loss,'--',lw=2, label='Validation loss',color='black')
plt.legend(loc='upper right')
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Mean squared error',fontsize=16)
plt.show()




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 7. Test the performance of the model prediction.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Load new data, which has different mode patterns from the training datasets.
result_dict=loadmat("xun_testing_1")
print("type of reuslt:",type(result_dict))
print("keys:",result_dict.keys()) 
psi_total = result_dict['Psi_nf1']
u_nf_total = result_dict['u_nf1']
p_nf_total = result_dict['P_nf1']
mm_test= result_dict['mm']
nn_test= result_dict['nn']


psi3_total = np.zeros([psi_total.shape[0],int(y_num[0,0]),int(x_num[0,0]),2])
u_nf3_total= np.zeros([psi_total.shape[0],int(y_num[0,0]),int(x_num[0,0]),2])
id=0
while id < psi_total.shape[0]:
    tmp=psi_total[id,:]
    tmp=tmp.reshape([int(y_num[0,0]),int(x_num[0,0])]) 
    psi3_total[id,:,:,0]=tmp.real
    psi3_total[id,:,:,1]=tmp.imag  
    tmp=u_nf_total[id,:]
    tmp=tmp.reshape([int(y_num[0,0]),int(x_num[0,0])])   
    u_nf3_total[id,:,:,0]=tmp.real
    u_nf3_total[id,:,:,1]=tmp.imag               
    id=id+1

# Display the testing figures    
id=0
while id < psi_total.shape[0]:
    psi=psi3_total[id,:,:,0]
    u_nf=u_nf3_total[id,:,:,0]
    fig2, ax2 = plt.subplots(constrained_layout=True)
    CS=ax2.contourf(zz,rr,u_nf, 20, cmap='RdGy')
    ax2.set_xlabel('x', fontsize=15,fontname='Times New Roman')
    ax2.set_ylabel('r', fontsize=15,fontname='Times New Roman')
    ax2.set_title('u-velocity', fontsize=18,fontname='Times New Roman')
    cbar = plt.colorbar(CS)
    plt.savefig('u_'+str(id)+'.jpg')
    plt.clf()    
    fig1, ax1 = plt.subplots(constrained_layout=True)
    CS1=ax1.contourf(zz,rr,psi, 20, cmap='RdGy')
    ax1.set_xlabel('x', fontsize=15,fontname='Times New Roman')
    ax1.set_ylabel('r', fontsize=15,fontname='Times New Roman')
    ax1.set_title('Acoustic potential', fontsize=18,fontname='Times New Roman')
    cbar = plt.colorbar(CS1)
    plt.savefig('psi_'+str(id)+'.jpg')
    plt.clf()
    id=id+1

#import time
#aaa=time.clock()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Prediction
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pred=model.predict(psi3_total[:,:,:,:]) #[np.newaxis])
#bbb=time.clock()
u_pred=pred 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Plot three figures: testing input; prediction; difference between input and prediction. 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pred_id=0
# Plot prediction 
bar_range = np.linspace(-2,2,15,endpoint=True)
fig1, ax1 = plt.subplots(constrained_layout=True)
CS1=ax1.contourf(zz,rr,u_pred[pred_id,:,:,0], bar_range, cmap='RdGy')
ax1.set_xlabel('x', fontsize=15,fontname='Times New Roman')
ax1.set_ylabel('r', fontsize=15,fontname='Times New Roman')
#C1=ax1.contour(zz,rr,u_pred[:,:],20)
#plt.clabel(C1,inline=True,fontsize=12)
ax1.set_title('Machine prediction', fontsize=18,fontname='Times New Roman')
cbar = plt.colorbar(CS1,ticks=bar_range)
plt.savefig('predu_chan2_unet10k_otanh'+str(pred_id)+'.jpg')

# Plot real result
fig2, ax2 = plt.subplots(constrained_layout=True)
CS2=ax2.contourf(zz,rr,u_nf3_total[pred_id,:,:,0], bar_range,  cmap='RdGy')
ax2.set_xlabel('x', fontsize=15,fontname='Times New Roman')
ax2.set_ylabel('r', fontsize=15,fontname='Times New Roman')
ax2.set_title('Wiener-Hopf', fontsize=18,fontname='Times New Roman')
cbar = plt.colorbar(CS2, ticks=bar_range)
plt.savefig('WHu_chan2_unet10k_otanh'+str(pred_id)+'.jpg')


bar_range = np.linspace(-0.2,0.2,15,endpoint=True)
# Plot real result
fig3, ax3 = plt.subplots(constrained_layout=True)
CS3=ax3.contourf(zz,rr,u_nf3_total[pred_id,:,:,0]-u_pred[pred_id,:,:,0], bar_range,  cmap='RdGy')
ax3.set_xlabel('x', fontsize=15,fontname='Times New Roman')
ax3.set_ylabel('r', fontsize=15,fontname='Times New Roman')
ax3.set_title('Difference', fontsize=18,fontname='Times New Roman')
cbar = plt.colorbar(CS3, ticks=bar_range)
plt.savefig('diff_chan2_unet10k_otanh'+str(pred_id)+'.jpg')
plt.show()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# End of the demonstration. 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



