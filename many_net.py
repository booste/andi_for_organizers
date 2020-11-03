#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#function to apply different nets on trajectories of different lengths
#it assumes equal spacing of the nets and that they are centered at 1/4 of the range
#it also assumes all nets take the same dimension, this can be changes
#works for 1d data
import numpy as np

   




     

                
def many_net_only_diff_cont_varc(nets,traj_set,centers,skip=[],min_tr=0,max_tr=1000):
    """takes as input list of networks, data set and the vector centers of where the different nets
    were trained on. 
    Returns an array of predictions.
    For each trajectory it uses a weighted average of the predictions made by the two networks
    that were trained on lengths closest to the one of the trajectory to be predicted.
    NB skip functionality is not worked out"""
    centers=np.asarray(centers)  
    n_nets=len(nets) #number of nets we can use

    di=[]
    
    # creating array with input dimension of the different networks in use 
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    
    # predicting each trajectory
    predictions_comb=[]
    for traj in traj_set:
        jj=len(traj)
        #choosing which net to use, based on trajectory length.
        #The chosen net is the one trained on trajectories that are closest in length (and shorter) to the trajectory to be predicted
        if jj<=centers[0]:
            k=0
        elif jj>np.max(centers):
            k=n_nets-1
        else:
            
            k=np.argmax(jj<np.asarray(centers))-1
       
        #cutting the trajectory to make it of length  multiple of dimensione used by net
        rl=int((jj-1)/di[k])*di[k]  
        
        #taking the increments of the trajectory
        traj=np.diff(traj)
        
        #normalizing trajectory
        
        sd=np.std(traj)
        if sd>0:
            traj=(traj-np.mean(traj))/sd
        else:
            traj=(traj-np.mean(traj))
            #print('traj len=',jj,'diff')
        
            
            
       
         #print(rl)
         
        # reshaped trajectory to fit network requirement

        rs_traj = np.asarray(traj[:rl]).reshape(1,int(rl/di[k]),di[k]) 
        #of the network length
        #print(len(traj),"chosen net",k)
        
        #predicting from the trajectory
        pr_b=nets[k].predict(rs_traj).flatten()
        
        #Use also the closest net from the right (longer than the traj to be predicted)
        #The 2 predictions will be combined by a weighted average
        #Unless the trajectory is shorter (longer) than the shortest (longest) net
        if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
            #distance between the first net used and the following one
            ran=centers[k+1]-centers[k]
            #distance between traj len (after cutting) and center of net used
            d=(rl+1-centers[k])/ran  
            if d>=0:
            

                rl_b=int((jj-1)/di[k+1])*di[k+1] 
                rs_traj_b = np.asarray(traj[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
    #                print("combine! length=",rl,
    #                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                
                #the final prediction is a weighted average between the preds of the nets closest to trajectory length
                pr_b=((1-d)*pr_b+d*pr_2b)

        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()



#function to split multi-dimensional trajectories into array with row for each dimension
def high_d(traj,d,thr=0):
    d=int(d)
    i = int(len(traj)/d)
    xvec = np.ones((d,i-1))
    traj=np.asarray(traj)
    for kk in range(d):
       # print(len(traj),i)
        
        x = np.diff(traj[kk*i:(kk+1)*i])  # separate x data

        sx=np.std(x)
        xvec[kk,:] = (x-np.mean(x)) / np.where(sx>thr,sx,1)   # normalize x data
    return xvec



def many_net_odcv_hd(nets,traj_set,centers,dim,thr=1e-12,skip=[],min_tr=0,max_tr=1000):
    """For multidimensional trajectories: it splits the data into one dimensional trajectories
    along each dimensions. It returns two arrays, the first one is the prediction made on 
    all different spatial dimension; the second array is the average
    of predictions over the different dimensions.
    Takes as input list of networks, data set and the vector centers of where the different nets
    were trained on, dim: spatial dimension of trajectories. thr is the threshold for the variance
    of the trajectory used in the normalization to avoid division by 0 for constant trajectories.
    For each trajectory it uses a weighted average of the predictions made by the two networks
    that were trained on lengths closest to the one of the trajectory to be predicted.
    NB skip functionality is not worked out
    The steps that are the same as in the function above are not fully commented
    """
    centers=np.asarray(centers)
    n_nets=len(nets) #number of nets we can use
    #sp=max_tr/n_nets  #length of range that on which each net will be used
    ##print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions=[]        #predictions for the various dimensions
    predictions_ave=[]    #predictiona averaged over dimensions
    tot_tt=len(traj_set)
    count=0
    for traj in traj_set:
        count=count+1
        print('traj',count,'/',tot_tt, end='\r')
        dim=int(dim)
        jj = int(len(traj)/dim)   #length of the trajectory
        #print(jj)
        #xvec = np.ones((d,jj-1))
        traj=np.asarray(traj)
        pr_ave=0
        
        #loops over the different dimensions
        for rr in range(dim):
           # #print(len(traj),i)

            x = np.diff(traj[rr*jj:(rr+1)*jj])  # separate  data for each dimension

            sx=np.std(x)
            x = (x-np.mean(x)) / np.where(sx>thr,sx,1)   #Normalize data
        #choosing which net to use
            if jj<=centers[0]:
                k=0
            elif jj>np.max(centers):
                k=n_nets-1
            else:

                k=np.argmax(jj<np.asarray(centers))-1
            #k=int((jj-min_tr)/sp)  #choosing which net to use
            ##print(jj)
            #print(k)
            rl=int((jj-1)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net

            #print(rl)
            rs_traj = np.asarray(x[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

            #of the network length
            ##print(len(traj),"chosen net",k)
            pr_b=nets[k].predict(rs_traj).flatten()
            #print(pr_b)
            if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
                #distance between the net used and the following one
                ran=centers[k+1]-centers[k]
                dist=(rl+1-centers[k])/ran   #distance between traj len (after cutting) and center of net used
                if dist>=0:


                    rl_b=int((jj-1)/di[k+1])*di[k+1] 
                    rs_traj_b = np.asarray(x[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
        #                #print("combine! length=",rl,
        #                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                    pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                    pr_b=((1-dist)*pr_b+dist*pr_2b)
                    #print(pr_b)
            
            #Progressively averages the predictions for each dimension
            pr_ave+=pr_b/dim
            
            #Appending the prediction for each dimension.
            #It will return a list of length d*traj
            #organized as in {{traj1x,traj2x,traj3x}}}
            predictions.append(pr_b)       
        
        #Once the loop over dimensions is over it appends the average prediction        
        predictions_ave.append(pr_ave)

    return np.asarray(predictions).flatten(), np.asarray(predictions_ave).flatten()





    
def many_net_only_diff_cont_varc_dim(nets,traj_set,centers,dim,skip=[],min_tr=0,max_tr=1000):
    """Fot networks trained on higher dimensions! Takes as input list of networks, data set and
    the vector centers of where the different nets
    were trained on. Also needs dimension of trajectory.
    The higher dimensional trajectory is given as a 1d array collating the different dimensions
    e.g. r=np.concatenate(x,y,z)
    NB skip functionality is not worked out"""
    centers=np.asarray(centers)
    n_nets=len(nets) #number of nets we can use
    #sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_comb=[]
    tot_tt=len(traj_set)
    count=0
    for traj in traj_set:
        count=count+1
        print('traj',count,'/',tot_tt,end='\r')
        jj=len(traj)            #length of trajectory times dimension
        js=int(jj/dim)          #length of trajectory
        #choosing which net to use
        if js<=centers[0]:
            k=0
        elif js>np.max(centers):
            k=n_nets-1
        else:
            
            k=np.argmax(js<np.asarray(centers))-1
        #k=int((jj-min_tr)/sp)  #choosing which net to use
#         print(jj)
#         print(js)
        
        
        #taking the diff and reshaping the trajectory
        
        thr=1e-10
        X=np.asarray(traj)
        r = X.reshape(1,dim,js) 
        r = np.diff(r,axis=2) 
        
        x = r[:,0,:]
        sx = np.std(x,axis=1)
        x = (x-np.mean(x,axis=1).reshape(len(x),1)) / np.where(sx>thr,sx,1).reshape(len(x),1)   # normalize x data
        
        for dm in range(1,dim):
            y = r[:,dm,:]
            sy = np.std(y,axis=1)
            y = (y-np.mean(y,axis=1).reshape(len(y),1)) / np.where(sy>thr,sy,1).reshape(len(y),1)   # normalize y data
            x = np.concatenate((x,y),axis=1)
        
        r=x
        #print(r.shape)
        rl=int((jj-dim)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
            
            
       
        #print(rl)
        rs_traj = np.asarray(r[:,:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
            #distance between the net used and the following one
            ran=centers[k+1]-centers[k]
            d=(js-centers[k])/ran   #distance between traj len (after cutting) and center of net used
           # print(js,centers[k],ran,d,'\n')
            if d>0:
            

                rl_b=int((jj-dim)/di[k+1])*di[k+1] 
                rs_traj_b = np.asarray(r[:,:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
    #                print("combine! length=",rl,
    #                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                pr_b=((1-d)*pr_b+d*pr_2b)

        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()

       



def many_net_only_diff_cont_varc_2d_4_3d(nets,traj_set,centers, skip=[],min_tr=0,max_tr=1000):
    """takes as input list of networks, data set and the vector centers of where the different nets
    were trained on.  Meant for 3d trajectories to be analyzed
    as averages of 2d trajectories. NB skip functionality is not worked out"""
    centers=np.asarray(centers)
    n_nets=len(nets) #number of nets we can use
    #sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    dim=3 #fixing 3d case
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_ave=[]
    tot_tt=len(traj_set)
    count=0
    for traj in traj_set:
        count=count+1
        print('traj',count,'/',tot_tt,end='\r')
        jj=len(traj)            #length of trajectory times dimension
        js=int(jj/dim)          #length of trajectory
        #print(js)
        #choosing which net to use
        if js<=centers[0]:
            k=0
        elif js>np.max(centers):
            k=n_nets-1
        else:
            
            k=np.argmax(js<np.asarray(centers))-1
        #k=int((jj-min_tr)/sp)  #choosing which net to use
#         print(jj)
#         print(js)
        
        
        #taking the diff and reshaping the trajectory
        
        thr=1e-10
        X=np.asarray(traj)
        r = X.reshape(1,dim,js) 
        r = np.diff(r,axis=2) 
        
        #normalizing data and creating list with 3 arrays one per each dimension (normalized)
        xyz=[]
        for dm in range(0,dim):
            y = r[:,dm,:]
            sy = np.std(y,axis=1)
            y = (y-np.mean(y,axis=1).reshape(len(y),1)) / np.where(sy>thr,sy,1).reshape(len(y),1)   # normalize y data
           # x = np.concatenate((x,y),axis=1)
            #print('normalized traj in dim',dm+1,'is',y[:,:5],'\n')
            xyz.append(y)
            
        #creating 2d trajectories using only 2 of the 3 dimensions (normalized)    
        xy=np.concatenate((xyz[0],xyz[1]),axis=1)
        xz=np.concatenate((xyz[0],xyz[2]),axis=1)
        yz=np.concatenate((xyz[1],xyz[2]),axis=1)

        xyz=[xy,xz,yz]
        #r=x
        #print(r.shape)
        rl=int((jj-dim-(js-1))/di[k])*di[k] #cutting the trajectory to fit to  multiple of dimensione used by net
        pr_b_ave=0
        for dm in range(0,dim):

            rs_traj = np.asarray(xyz[dm][:,:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement
#             print('dim=',dm+1,xyz[dm][:,:5])
#             print(xyz[dm][:,js-1:js-1+5])
        #of the network length
        #print(len(traj),"chosen net",k)
            pr_b=nets[k].predict(rs_traj).flatten()

            if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
                #distance between the net used and the following one
                ran=centers[k+1]-centers[k]
                d=(js-centers[k])/ran   #distance between traj len (after cutting) and center of net used
               # print(js,centers[k],ran,d,'\n')

                if d>=0:


                    rl_b=int((jj-dim-(js-1))/di[k+1])*di[k+1] 


                  
                    rs_traj_b = np.asarray(xyz[dm][:,:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
        #                print("combine! length=",rl,
        #                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                    pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                    pr_b=((1-d)*pr_b+d*pr_2b)
           #print('prediction',pr_b,'\n')
            pr_b_ave=pr_b_ave+pr_b/dim
            #print('ave pred',pr_b_ave,'\n')
        predictions_ave.append(pr_b_ave)
        
    return np.asarray(predictions_ave).flatten()
