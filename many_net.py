#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#function to apply different nets on trajectories of different lengths
#it assumes equal spacing of the nets and that they are centered at 1/4 of the range
#it also assumes all nets take the same dimension, this can be changes
#works for 1d data
import numpy as np
def many_net(nets,traj_set,min_tr=0,center=25,max_tr=1000,comb=True):
    """takes as input list of networks, data set and the vector di 
    of the dimension of the data the different networks work on """
    n_nets=len(nets) #number of nets we can use
    sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    predictions_comb=[]
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    #print(di)
    for traj in traj_set:
        #normalizing trajectory
        traj=(traj-np.mean(traj))/np.std(traj)
        
        jj=len(traj)
        k=int((jj-min_tr)/sp)  #choosing which net to use
        #print(jj)
        rl=int(jj/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
        #print(rl)
        rs_traj = np.asarray(traj[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if comb==True:
            if ((rl-k*sp>sp/2)and(k<n_nets-1)):
                rl_b=int(jj/di[k+1])*di[k+1] 
                rs_traj_b = np.asarray(traj[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
                #print("combine! length=",rl,
                #      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                pr_b=(pr_b+pr_2b)/2
        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()
     
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#function to apply different nets on trajectories of different lengths
#it assumes equal spacing of the nets and that they are centered at 1/4 of the range
#it also assumes all nets take the same dimension, this can be changes
#works for 1d data
def many_net_diffv(nets,traj_set,diffv,min_tr=0,center=25,max_tr=1000,comb=True):
    """takes as input list of networks, data set and the vector di 
    of the dimension of the data the different networks work on """
    n_nets=len(nets) #number of nets we can use
    sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_comb=[]
    for traj in traj_set:
        jj=len(traj)
        k=int((jj-min_tr)/sp)  #choosing which net to use
        #print(jj)
        if diffv[k]==True:
            rl=int((jj-1)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
            traj2=np.diff(traj)
        
        #normalizing trajectory
            sd2 = np.std(traj2)
            if sd2>0:
                
                traj2=(traj2-np.mean(traj2))/sd2
            #print('traj len=',jj,'diff')
            else:
                traj2=(traj2-np.mean(traj2))
                
            
            
        else:
            
            rl=int(jj/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
       
        
        #normalizing trajectory
            sd=np.std(traj)
            if sd>0:
                
                traj2=(traj-np.mean(traj))/sd
            else:
                traj2=(traj-np.mean(traj))
        
         #print(rl)
        rs_traj = np.asarray(traj2[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if comb==True:
            
            if ((rl-k*sp>sp/2)and(k<n_nets-1)):
                kp=k+1
                
                if diffv[kp]==True:
                            
                    rl_b=int((jj-1)/di[kp])*di[kp]  #cutting the trajectory to fit to  multiple of dimensione used by net
                    traj2=np.diff(traj)
        
        #normalizing trajectory
                    sd2=np.std(traj2)
                    if sd2>0:
                        traj2=(traj2-np.mean(traj2))/sd2
                        
                    else:
                       traj2=(traj2-np.mean(traj2)) 
            #print('traj len=',jj,'diff')
        
            
            
                else:
            
                    rl_b=int(jj/di[kp])*di[kp]  #cutting the trajectory to fit to  multiple of dimensione used by net
       
        
        #normalizing trajectory
                    sd=np.std(traj)
                    if sd>0:
                        
                        traj2=(traj-np.mean(traj))/sd
                    else:
                        traj2=(traj-np.mean(traj))
                        
                rs_traj_b = np.asarray(traj2[:rl_b]).reshape(1,int(rl_b/di[kp]),di[kp])
                #print("combine! length=",rl,
                #      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[kp].predict(rs_traj_b).flatten()
                pr_b=(pr_b+pr_2b)/2

        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()
     

def many_net_only_diff(nets,traj_set,skip=[],min_tr=0,center=25,max_tr=1000,comb=True):
    """takes as input list of networks, data set and the vector di 
    of the dimension of the data the different networks work on """
    n_nets=len(nets) #number of nets we can use
    sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_comb=[]
    for traj in traj_set:
        jj=len(traj)
        k=int((jj-min_tr)/sp)  #choosing which net to use
        #print(jj)
     
        rl=int((jj-1)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
        traj=np.diff(traj)
        
        #normalizing trajectory
        sd=np.std(traj)
        if sd>0:
            traj=(traj-np.mean(traj))/sd
        else:
            traj=(traj-np.mean(traj))
            #print('traj len=',jj,'diff')
        
            
            
       
         #print(rl)
        rs_traj = np.asarray(traj[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if ((comb==True) and np.isin(k,skip,invert=True) ):
            
            if ((rl-k*sp>sp/2)and(k<n_nets-1)):
                rl_b=int((jj-1)/di[k+1])*di[k+1] 
                rs_traj_b = np.asarray(traj[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
#                print("combine! length=",rl,
#                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[k+1].predict(rs_traj_b).flatten()
                pr_b=(pr_b+pr_2b)/2

        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()
     

                
def many_net_only_diff_cont(nets,traj_set,skip=[],min_tr=0,center=25,max_tr=1000):
    """takes as input list of networks, data set and the vector di 
    of the dimension of the data the different networks work on """
    n_nets=len(nets) #number of nets we can use
    sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_comb=[]
    for traj in traj_set:
        jj=len(traj)
        k=int((jj-min_tr)/sp)  #choosing which net to use
        #print(jj)
        
        rl=int((jj-1)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
        d=(rl+1-(k*sp+center))/sp   #distance between traj len (after cutting) and center of net used
        traj=np.diff(traj)
        
        #normalizing trajectory
        sd=np.std(traj)
        if sd>0:
            traj=(traj-np.mean(traj))/sd
        else:
            traj=(traj-np.mean(traj))
            #print('traj len=',jj,'diff')
        
            
            
       
         #print(rl)
        rs_traj = np.asarray(traj[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if ((d>=0) and (k<n_nets-1) and np.isin(k,skip,invert=True) ):
            

            rl_b=int((jj-1)/di[k+1])*di[k+1] 
            rs_traj_b = np.asarray(traj[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
#                print("combine! length=",rl,
#                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
            pr_2b=nets[k+1].predict(rs_traj_b).flatten()
            pr_b=((1-d)*pr_b+d*pr_2b)

        predictions_comb.append(pr_b)
        
    return np.asarray(predictions_comb).flatten()
     

                
def many_net_only_diff_cont_varc(nets,traj_set,centers,skip=[],min_tr=0,max_tr=1000):
    """takes as input list of networks, data set and the vector centers of where the different nets
    were trained on. NB skip functionality is not worked out"""
    centers=np.asarray(centers)
    n_nets=len(nets) #number of nets we can use
    #sp=max_tr/n_nets  #length of range that on which each net will be used
    #print(sp)
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    predictions_comb=[]
    for traj in traj_set:
        jj=len(traj)
        #choosing which net to use
        if jj<=centers[0]:
            k=0
        elif jj>np.max(centers):
            k=n_nets-1
        else:
            
            k=np.argmax(jj<np.asarray(centers))-1
        #k=int((jj-min_tr)/sp)  #choosing which net to use
        #print(jj)
        
        rl=int((jj-1)/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
        traj=np.diff(traj)
        
        #normalizing trajectory
        
        sd=np.std(traj)
        if sd>0:
            traj=(traj-np.mean(traj))/sd
        else:
            traj=(traj-np.mean(traj))
            #print('traj len=',jj,'diff')
        
            
            
       
         #print(rl)
        rs_traj = np.asarray(traj[:rl]).reshape(1,int(rl/di[k]),di[k]) # reshaped trajectory to fir network requirement

        #of the network length
        #print(len(traj),"chosen net",k)
        pr_b=nets[k].predict(rs_traj).flatten()
        
        if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
            #distance between the net used and the following one
            ran=centers[k+1]-centers[k]
            d=(rl+1-centers[k])/ran   #distance between traj len (after cutting) and center of net used
            if d>=0:
            

                rl_b=int((jj-1)/di[k+1])*di[k+1] 
                rs_traj_b = np.asarray(traj[:rl_b]).reshape(1,int(rl_b/di[k+1]),di[k+1])
    #                print("combine! length=",rl,
    #                      "chosen net=",k,"distance between chosen net and traj",rl-k*sp)
                pr_2b=nets[k+1].predict(rs_traj_b).flatten()
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
        xvec[kk] = (x-np.mean(x)) / np.where(sx>thr,sx,1)   # normalize x data
    return xvec



def many_net_odcv_hd(nets,traj_set,centers,dim,thr=1e-12,skip=[],min_tr=0,max_tr=1000):
    """takes as input list of networks, data set and the vector centers of where the different nets
    were trained on. NB skip functionality is not worked out"""
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
    for traj in traj_set:
        dim=int(dim)
        jj = int(len(traj)/dim)   #length of the trajectory
        #print(jj)
        #xvec = np.ones((d,jj-1))
        traj=np.asarray(traj)
        pr_ave=0
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
            
            pr_ave+=pr_b/dim
            predictions.append(pr_b)       #NB it will return a list of length d*traj
                                                #organized as in {{traj1x,traj2x,traj3x}}}
                
        predictions_ave.append(pr_ave)

    return np.asarray(predictions).flatten(), np.asarray(predictions_ave).flatten()