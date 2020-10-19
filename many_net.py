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
            traj2=(traj2-np.mean(traj2))/np.std(traj2)
            #print('traj len=',jj,'diff')
        
            
            
        else:
            
            rl=int(jj/di[k])*di[k]  #cutting the trajectory to fit to  multiple of dimensione used by net
       
        
        #normalizing trajectory
            traj2=(traj-np.mean(traj))/np.std(traj)
        
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
                    traj2=(traj2-np.mean(traj2))/np.std(traj2)
            #print('traj len=',jj,'diff')
        
            
            
                else:
            
                    rl_b=int(jj/di[kp])*di[kp]  #cutting the trajectory to fit to  multiple of dimensione used by net
       
        
        #normalizing trajectory
                    traj2=(traj-np.mean(traj))/np.std(traj)
                
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
        traj=(traj-np.mean(traj))/np.std(traj)
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
        traj=(traj-np.mean(traj))/np.std(traj)
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
     

                
