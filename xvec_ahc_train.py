#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:51:43 2019

@author: prachi singh 
@email: prachisingh@iisc.ac.in 

This code is for DNN training 
Explained in paper:
P. Singh, S. Ganapathy, Deep Self-Supervised Hierarchical Clustering for Speaker Diarization, Interspeech, 2020

Check main function: train_with_threshold , to run for different iterations
"""

import os
import sys
import numpy as np
import random
import pickle
import subprocess
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from models_train_ahc import weight_initialization,Deep_Ahc_model
import torch.utils.data as dloader
from arguments import read_arguments as params
from pdb import set_trace as bp
sys.path.insert(0,'services/')
import kaldi_io
import services.agglomerative as ahc
from services.path_integral_clustering import PIC_ami,PIC_org_threshold,PIC_org, PIC_callhome, PIC_callhome_threshold, PIC_ami_threshold
sys.path.insert(0,'tools_diar/steps/libs')

# read arguments
opt = params()
#select device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid

# torch.manual_seed(777)  # reproducibility



loss_lamda = opt.alpha
dataset=opt.dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

# Model defined here
def normalize(system):
     # to make zero mean and unit variance
        my_mean = np.mean(system)
        my_std = np.std(system)
        system = system-my_mean
        system /= my_std
        return system

def compute_affinity_loss(output,cluster,lamda):

    mylist = np.arange(len(cluster))
    # print('mylist:',mylist)
    loss=0.0
    biglamda=0.0
    for k,c in enumerate(cluster):
        for i in range(len(c)-1):
            for j in range(i+1,len(c)):
                nlist=np.delete(mylist,k,0)
                # bp()
                try:
                    ind=np.random.choice(nlist)
                except:
                    bp()
                b = cluster[ind]
                # bp()
                bi = i % len(b)
                # min(i,abs(i-len(b)))
                loss += -output[c[i],c[j]]+ lamda*output[c[i],b[bi]]+ lamda*output[c[j],b[bi]]
                biglamda +=1

    return loss/biglamda
def mostFrequent(arr, n):

    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res


class Deep_AHC:
    def __init__(self,data,pldamodel,fname,reco2utt,xvecdimension,model,optimizer,n_prime,writer=None):
        self.reco2utt = reco2utt
        self.xvecdimension = xvecdimension
        self.model = model
        self.optimizer = optimizer
        self.n_prime = n_prime
        self.fname = fname
        self.final =0
        self.forcing_label = 0
        self.results_dict={}
        self.pldamodel = pldamodel
        self.data = data
        self.lamda = 0.0
        self.K = 30
        self.z = 0.1
       
       

    def write_results_dict(self, output_file):
        """Writes the results in label file"""
        f = self.fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = self.results_dict[f]
        meeting_name = f
        reco = self.reco2utt.split()[0]
        utts = self.reco2utt.rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=0
        segmentsfile = opt.segments+'/'+f+'.segments'
        python = opt.which_python
      
        cmd = '{} tools_diar/diarization/make_rttm.py --rttm-channel 0 {} {}/{}.labels {}/{}.rttm' .format(python,segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)
    

    def compute_score(self,rttm_gndfile,rttm_newfile,outpath,overlap):
      fold_local='services/'
      scorecode='score.py -r '
     
      # print('--------------------------------------------------')
      if not overlap:

          cmd=opt.which_python +' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' --ignore_overlaps --collar 0.25 -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      else:
          cmd=opt.which_python + ' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      # print('----------------------------------------------------')
      # subprocess.check_call(cmd,stderr=subprocess.STDOUT)
      # print('scoring ',rttm_gndfile)
      bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      output=subprocess.check_output(bashCommand,shell=True)
      return float(output.decode('utf-8').rstrip())
      # output = subprocess.check_output(['bash','-c', bashCommand])

    def compute_loss(self,A,minibatch,lamda):
        loss = 0.0
        weight = 1

        for m in minibatch:
            loss += -weight*A[m[0],m[1]]+lamda*(A[m[0],m[2]]+A[m[1],m[2]])+ 1.0
        # print('sum loss : ',loss)
        return loss/len(minibatch)

    def compute_minibatches(self,A,cluster,labels,mergeind=[],cleanind = []):
        triplets = []
        hard = 0
        random_sample = 1
        multiple = 0
        for ind,k in enumerate(cluster):
            neg = np.where(labels!=ind)[0]
            for i,a in enumerate(k[:-1]):                    
                for p in k[i+1:]:
                    Aavg = (A[a,neg]+A[p,neg])/2.0
                   
                    if hard:                            
                        neg_ind = np.argmax(Aavg)
                        fetch_negatives = neg[neg_ind]
                        triplets.append([a,p,fetch_negatives])
                    if random_sample:
                        max_10 = random.randint(0,len(Aavg)-1)
                        max_neg = min(max_10,len(Aavg)-1)
                        fetch_negatives = neg[max_neg]
                        triplets.append([a,p,fetch_negatives]) 
                    if multiple:
                        max_neg = np.random.randint(1, len(Aavg), size=(10,))
                        #neg_indices = np.argsort(Aavg,axis=None)[::-1][max_neg-1]
                        fetch_negatives = neg[max_neg]
                        for n in fetch_negatives:
                            triplets.append([a,p,n])

        
        random.shuffle(triplets)
        if len(triplets)==0:
            ValueError("No triplets generated!")
        triplets = np.array(triplets)
        N = len(triplets)
        N1=0

        num_batches = min(opt.N_batches,N)
        N1 = N -(N % num_batches)
        batchsize = int(N1/num_batches)
        print('batchsize:',batchsize)
        minibatches = triplets[:N1].reshape(-1,batchsize,3)
        
        return minibatches,batchsize
    
    def compute_minibatches_train(self,period,A,cluster,labels,overlap_ind=[],clusterlen = []):
        triplets = []
        random_sample = 0
        multiple = 1
        if len(overlap_ind)>0:
            last_ind = -1
        else:
            last_ind = None
        trainlabels = np.arange(len(labels))
        anch_count_full = int(np.mean(clusterlen))
        max_triplets = (anch_count_full-1)**2
        pos_count = anch_count_full
        print('anch_count_full:',anch_count_full)

        triplets_cluster = []
        triplets_cluster_size = np.zeros((len(cluster[:last_ind]),),dtype=int)
        ind_count = 0
        for ind,k in enumerate(cluster[:last_ind]):
           # if len(k)<10:
           #     continue 
           triplets = []
           train_neg = trainlabels[np.where(labels[trainlabels]!=ind)[0]]
           traink = k
           anch_count = min(anch_count_full,len(traink))
           # bp()
           possible_triplets = (anch_count-1)**2
           num_negatives = int(max_triplets/possible_triplets) # find number of negatives needed to keep balance between small clusters vs large
           for i,a in enumerate(traink[:anch_count-1]):      # mining number of triplets based on smallest/largest cluster              
               for p in traink[i+1:anch_count]:
                   if len(train_neg)!=0:
                     
                       if random_sample:
                           max_10 = random.randint(0,len(train_neg))
                           max_neg = min(max_10,len(train_neg)-1)
                           fetch_negatives = train_neg[max_neg]
                           triplets.append([a,p,fetch_negatives]) 
                       if multiple:                             
                           minsize=min(num_negatives,len(train_neg))
                           max_neg = np.random.randint(0, len(train_neg)-1, size=(minsize,))
                           #neg_indices = np.argsort(Aavg,axis=None)[::-1][max_neg-1]
                           fetch_negatives = train_neg[max_neg]
                           for n in fetch_negatives:
                              triplets.append([a,p,n])
                       
                   else:
                       triplets.append([a,p,0])
        triplets_cluster.append(triplets)
        triplets_cluster_size[ind_count] = len(triplets)
        ind_count = ind_count + 1
        triplets_cluster_size = triplets_cluster_size[:ind_count]

        triplets_cluster = np.array(triplets_cluster)
        N1=0
        batches = 1
        if batches >= min(triplets_cluster_size):
            batches = int(batches/2)
        num_batches = min(batches,min(triplets_cluster_size))
        print('num_batches:',num_batches)

        minibatches = []
        # bp()
        for k,triplets in enumerate(triplets_cluster):
                # cluster_batchsize = batchratio[k]              
                N = len(triplets)
                random.shuffle(triplets)
                N1 = N -(N % num_batches)
                cluster_batchsize = int(N1/num_batches)              
                
                # N1 = int(N1/cluster_batchsize)*cluster_batchsize
                print('cluster_batchsize:',cluster_batchsize)
                triplets_n1 = np.array(triplets[:N1])
                minibatches.append(triplets_n1.reshape(num_batches,-1,3))
                print('minibatch shape:',minibatches[k].shape)

        minibatches_full = []
        for i in range(num_batches):
            for k,triplets in enumerate(triplets_cluster):
                # bp()
                if k==0:
                    minibatches_full.append(minibatches[k][i].tolist())
                else:
                    minibatches_full[i].extend(minibatches[k][i].tolist())
            print('batchsize of batch {}: {}'.format(i,len(minibatches_full[i])))
        batchsize = len(minibatches_full[0])
        print('batchsize:',batchsize)
        
        print("total triplets : ",len(minibatches_full)*batchsize)
        # return minibatches_full,batchsize
        return np.array(minibatches_full)
    
    def compute_minibatches_train_valid(self,A,cluster,labels,overlap_ind=[],clusterlen = []):
        triplets = []
        tripletsval = []
        
        # labels= labels[np.arange(len(labels))!=cluster[-1]] 
        overlap_label = len(clusterlen)-1  
        clean_ind = np.where(labels!=overlap_label)[0]
        # labels = labels [clean_ind]    
        trainlabels = []
        val_labels = []
        train = 0.8

        random_sample = 0
        random_sample_val = 1
        multiple = 1
      
        if len(overlap_ind)>0:
            last_ind = -1
        else:
            last_ind = None

        for i,k in enumerate(cluster[:last_ind]):
            traink = k[:int(len(k)*train)]
            valk = k[int(len(k)*train):]
            
            trainlabels.extend(traink)
            val_labels.extend(valk)
        trainlabels = np.array(trainlabels)
        val_labels = np.array(val_labels)

        anch_count_full = int(train * np.mean(clusterlen))
        max_triplets = (anch_count_full-1)**2
        print('anch_count_full:',anch_count_full)
        triplets_cluster = []
        triplets_cluster_size = np.zeros((len(cluster[:last_ind]),),dtype=int)
        ind_count = 0
        for ind,k in enumerate(cluster[:last_ind]):
           if len(k)<10:
               continue 

           triplets = []
           train_neg = trainlabels[np.where(labels[trainlabels]!=ind)[0]]
           # neg = trainlabels[neg]
           # train_neg = neg[neg == np.array(trainlabels)]
           
           traink =  k[:int(len(k)*train)]
           valk = k[int(len(k)*train):]
           anch_count = min(anch_count_full,len(traink))
           # bp()
           possible_triplets = (anch_count-1)**2
           num_negatives = int(max_triplets/possible_triplets) # find number of negatives needed to keep balance between small clusters vs large
           for i,a in enumerate(traink[:anch_count-1]):      # mining number of triplets based on smallest/largest cluster              
               for p in traink[i+1:anch_count]:

                   if len(train_neg)!=0:
                     
                       if random_sample:
                           max_10 = random.randint(0,len(train_neg))
                           max_neg = min(max_10,len(train_neg)-1)
                           fetch_negatives = train_neg[max_neg]
                           triplets.append([a,p,fetch_negatives]) 
                       if multiple:                             
                           minsize=min(num_negatives,len(train_neg))
                           max_neg = np.random.randint(0, len(train_neg)-1, size=(minsize,))
                           #neg_indices = np.argsort(Aavg,axis=None)[::-1][max_neg-1]
                           fetch_negatives = train_neg[max_neg]
                           for n in fetch_negatives:
                              triplets.append([a,p,n])
                       
                   else:
                       triplets.append([a,p,0])
           val_neg = val_labels[np.where(labels[val_labels]!=ind)[0]]
           # val_neg = neg[neg == np.array(val_labels)]                    
           for i,a in enumerate(valk[:-1]):                    
               for p in valk[i+1:]:
                   # Aavg = (A[a,neg]+A[p,neg])/2.0
                   # if (len(Aavg))<=2:
                   #     bp()
                   if len(val_neg)!=0:
                       if random_sample_val:
                           max_10 = random.randint(0,len(val_neg))
                           max_neg = min(max_10,len(val_neg)-1)
                           fetch_negatives = val_neg[max_neg]
                           tripletsval.append([a,p,fetch_negatives]) 
                   
                   else:
                       tripletsval.append([a,p,0])
           triplets_cluster.append(triplets)
           triplets_cluster_size[ind_count] = len(triplets)
           ind_count = ind_count + 1
        triplets_cluster_size = triplets_cluster_size[:ind_count]

        random.shuffle(tripletsval)
        tripletsval = np.array(tripletsval)

        triplets_cluster = np.array(triplets_cluster)
        # N = len(triplets)
        N1=0

        batches = opt.N_batches
        if batches >= min(triplets_cluster_size):
            batches = int(batches/2)
        num_batches = min(batches,min(triplets_cluster_size))
        print('num_batches:',num_batches)
       
        minibatches = []
        for k,triplets in enumerate(triplets_cluster):
                # cluster_batchsize = batchratio[k]              
                N = len(triplets)
                random.shuffle(triplets)
                N1 = N -(N % num_batches)
                cluster_batchsize = int(N1/num_batches)              
                print('cluster_batchsize:',cluster_batchsize)
                triplets_n1 = np.array(triplets[:N1])
                minibatches.append(triplets_n1.reshape(num_batches,-1,4))
                print('minibatch shape:',minibatches[k].shape)

        minibatches_full = []
        for i in range(num_batches):
            for k,triplets in enumerate(triplets_cluster):
                # bp()
                if k==0:
                    minibatches_full.append(minibatches[k][i].tolist())
                else:
                    minibatches_full[i].extend(minibatches[k][i].tolist())
            
        batchsize = len(minibatches_full[0])
        print('batchsize:',batchsize)

        print("total triplets : ",len(minibatches_full)*batchsize)
        return minibatches_full,batchsize,tripletsval


    def compute_cluster(self,labels):
        unifull = np.unique(labels)
        ind = []
        for i,val in enumerate(unifull):
            ind.append((np.where(labels==val)[0]).tolist())
        return ind

    def dataloader_from_list(self):
        reco2utt = self.reco2utt
        D = self.xvecdimension

        channel = 1

        reco2utt=reco2utt.rstrip()
        f=reco2utt.split()[0]

        utts = reco2utt.split()[1:]
        
        if os.path.isfile(opt.xvecpath+f+'.npy'):
            system = np.load(opt.xvecpath+f+'.npy')

        else:
            arkscppath=opt.xvecpath+'xvector.scp'
            xvec_dict= { key:mat for key,mat in kaldi_io.read_vec_flt_scp(arkscppath) }
            system = np.empty((len(utts),D))
            for j,key in enumerate(utts):
                system[j] = xvec_dict[key]
            if not os.path.isdir(opt.xvecpath):
                os.makedirs(opt.xvecpath)
            np.save(opt.xvecpath+f+'.npy',system)

        x1_array=system[np.newaxis]
        data_tensor = torch.from_numpy(x1_array).float()
        self.data = data_tensor

    def train_with_AHCthreshold(self,model_init):
        """
        train the network using triplet loss
        Threshold range : Decide intial number of clusters N0
        th = [0.0,0.1,0.2]
        ##############################
        Set following parameters here
        -----------------------------
        th_count : index of "th" array , selects the starting threshold (default: 1)
        stop_period: How many iterations to run (default: 1 i.e train for 1 iteration and then go till N*)
     
        ###############################

        saves the model,
        score matrix
        partial score matrix using previous labels

        period : iteration number from period 

        Parameters
        ----------
        weight initialization

        Returns
        -------
        None.

        """
       
        th_count = 0
        th = [0.0,0.1,0.2]
        set_dist = th[th_count] # setting threshold = 0.0
        stop_period  = max(1,th_count)


        model = self.model
        optimizer = self.optimizer
        alpha =loss_lamda
        count = 0
        f = self.fname
        data = self.data
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        
        inpdata =  data.float().to(device)
        nframe = data.size()[1] 
        
        n_prime = self.n_prime
        print('starting cluster: ',nframe)

        max_spks = n_prime
        period0len = n_prime
       
        stop_pt = nframe - max_spks # stop at 10 clusters
        period=0

        current_lr = opt.lr
        t=0
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)

        while period < stop_period: # only 2 steps, increase for more iterations
           
            if period==0:
                model.eval()
               
                PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
                model.init_weights(PCA_transform)

                output_model= model(inpdata)
                # output_model = model.compute_plda_affinity_matrix(inpdata_init)
                output_model1 = output_model.detach().cpu().numpy()[0]
                output_model = output_model1.copy()
                cosinefold = 'cosine_pca_baseline/{}_scores/cosine_scores/'.format(dataset)
                cosinefile = '{}/{}.npy'.format(cosinefold,f)
                
                if not os.path.isdir(cosinefold):
                    os.makedirs(cosinefold)
                
                if not os.path.isfile(cosinefile):
                    np.save(cosinefile,output_model)                

                clusterlen_old = clusterlen_feed.copy()
                labelfull_old = labelfull_feed.copy()
                n_clusters = max(period0len,n_prime)             

                # generate clusters using threshold set_threshold and n_clusters whichever reaches first
                myahc =ahc.clustering(n_clusters,clusterlen_feed, self.lamda,labelfull_feed,dist=set_dist)
                labelfull,clusterlen,_ = myahc.my_clustering_full(output_model)
                cluster = self.compute_cluster(labelfull)
                n_clusters = len(clusterlen)
                period0len = n_clusters
                               
                t = t+ nframe-n_clusters
            else:
               
                model.eval()
                output = model(inpdata)
               
                output = output.cpu().detach().numpy()[0]
                output_new = output.copy()
                nframe1 = period0len
                # use new model for period = 1 else use previous labels and then proceed 
                if period !=1: 
                    clusterlen_old = clusterlen.copy()
                    labelfull_old = labelfull.copy()
                    unifull = np.unique(labelfull_old)

                    clusterlist=[]
                    for val in unifull:
                        ind=np.where(labelfull_old==val)[0]
                        clusterlist.append(ind[0])
                        avg=np.sum(output_new[ind],axis=0)
                        output_new[ind[0]]=avg
                        output_new[:,ind[0]]=avg
                    output_new = output_new[np.ix_(clusterlist,clusterlist)]
                    nframe1 = output_new.shape[-1]
                n_clusters_old = n_clusters
                n_clusters = max(n_prime,max_spks)
                th_count = th_count-1
                set_dist = th[th_count]
                # use threshold in second training
                myahc =ahc.clustering(n_clusters, clusterlen_old,self.lamda,labelfull_old,dist=set_dist)
                labelfull,clusterlen,mergeind = myahc.my_clustering_full(output_new)
                
                cluster = self.compute_cluster(labelfull)
                alpha  = alpha + 0.1
                print('merging to ... ',min(nframe1-len(clusterlen),nframe1-n_prime))
                
                n_clusters = len(clusterlen)
                model.init_weights(PCA_transform)
               
                output = model(inpdata)
               
                output = output.detach().cpu().numpy()[0]
                if n_clusters==max_spks:
                    t=stop_pt
                else:
                    t = nframe - n_clusters

            
            if period == 0:
              
                minibatches,batchsize = self.compute_minibatches(output_model1, cluster, labelfull)

               
                model.eval()
                output = model(inpdata)
               
                output = output.cpu().detach().numpy()[0]
                unifull = np.unique(labelfull)
                output_new = output.copy()
                clusterlist=[]
                for val in unifull:
                    ind=np.where(labelfull==val)[0]
                    clusterlist.append(ind[0])
                    avg=np.sum(output_new[ind],axis=0)
                    output_new[ind[0]]=avg
                    output_new[:,ind[0]]=avg
                output_new = output_new[np.ix_(clusterlist,clusterlist)]

                print('PCA intialisation with n_clusters:',n_clusters)
                valcluster,val_label = self.validate(output, count,n_clusters,clusterlen_feed,labelfull_feed,1)

                valclusterlen = []
                for c in valcluster:
                    valclusterlen.append(len(c))
                print('clusterlen: ',valclusterlen)
                count +=1
                per_loss = opt.eta
                avg_loss = self.compute_loss(output, minibatches.reshape(-1,3), loss_lamda)
                print("\n[epoch %d] avg_loss: %.3f" % (0,avg_loss))
               
            else:
               
                minibatches,batchsize = self.compute_minibatches(output, cluster, labelfull,mergeind = mergeind)

                
                print('\n-------------------------------------------------------')

                print('Baseline Cosine DER with n_clusters:',n_clusters)
                valcluster,val_label=self.validate(output_model, count,n_clusters,clusterlen_feed,labelfull_feed,2)
                count +=1
                print('Before training DER with n_clusters:',n_clusters)
                _,_=self.validate(output_new, count,n_clusters,clusterlen_old,labelfull_old,0)  # starting from previous merge
                count +=1

            for epoch in range(opt.epochs):

                model.train()

                model.zero_grad()
                self.optimizer.zero_grad()
                out_train = model(inpdata)
               
                triplet_avg_loss = self.compute_loss(out_train[0],minibatches.reshape(-1,3),alpha)
                
                tot_avg_loss = triplet_avg_loss
                print("\n[epoch %d]  triplet_avg_loss: %.5f " % (epoch+1,triplet_avg_loss))
                if epoch == 0:
                    avg_loss = tot_avg_loss
                if tot_avg_loss < per_loss*avg_loss:
                    break
                tot_avg_loss.backward()
                self.optimizer.step()
               

            print('At t=',t,' clusters now:',len(cluster))

           
            period = period + 1

        print('Pre-training DER with N*:',n_prime)
        _,_=self.validate(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        count +=1
        self.final = 1
        model.eval()
        output1 = model(inpdata)
        # output1 = model.compute_plda_affinity_matrix(model(inpdata))
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        print('System DER with N*:',n_prime)
    
        valcluster,val_label=self.validate(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        count +=1
        print('Saving learnt parameters')
        matrixfold = "%s/cosine_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
       
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
             pickle.dump(savedict,sf)
                

        print('\n-------------Saving model------------------------------------------')
        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')
        
        return model

    def train_with_amiPIC_threshold(self,model_init,pretrain=0):
        """
        Train the SSC using N* clusters using PIC

        Parameters
        ----------
       Initial model

        Returns
        -------
        None.

        """
        model = self.model
        optimizer = self.optimizer
        alpha = loss_lamda # final stage
        count = 0
        f = self.fname
        data = self.data
        
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        count +=1
        inpdata =  data.float().to(device)
        nframe = data.size()[1]

        n_prime = self.n_prime
        print('starting cluster: ',nframe)

        phi_range = [0.5,0.6,0.7]
        phi_count = 1
        
        max_spks = n_prime
        period0len = n_prime
        period = 0
        current_lr = opt.lr
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)

        model.eval()
        stop_period = 1
        while period < stop_period:
            if not pretrain or period ==0:
                # initialize filewise PCA using svd                  
                PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
                model.init_weights(PCA_transform)
                output_model,_ = model(inpdata)
                # output_model = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata)
                output_model1 = output_model.detach().cpu().numpy()[0]
                output_model = output_model1.copy()
                
                cosinefold = 'cosine_pca_baseline/{}_scores/cosine_scores/'.format(dataset)
                # # cosinefold = 'scores_plda_new/{}_scores/'.format(dataset,f)
                cosinefile = '{}/{}.npy'.format(cosinefold,f)
                if not os.path.isdir(cosinefold):
                    os.makedirs(cosinefold)
               
                if not os.path.isfile(cosinefile):
                    np.save(cosinefile,output_model) 
            else:
                model = model_init
                output_model,_ = model(inpdata)
                output_model1 = output_model.detach().cpu().numpy()[0]
                output_model = output_model1.copy()
            labelfull_feed=np.arange(output_model.shape[0])
            clusterlen_feed=[1]*len(labelfull_feed)    
           
    
            clusterlen_old = clusterlen_feed.copy()
            labelfull_old = labelfull_feed.copy()
            n_clusters = max(period0len,n_prime)
       
            distance_matrix = (output_model+1)/2
            mypic =PIC_org_threshold(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),phi_range[phi_count],K=self.K,z=self.z) 
            labelfull,clusterlen = mypic.gacCluster() 
            n_clusters = len(clusterlen)
            cluster = self.compute_cluster(labelfull)
            phi_count = min(0,phi_count - 1)


        minibatches,batchsize,tripletsval = self.compute_minibatches_train_valid(output_model1, cluster, labelfull,clusterlen=clusterlen)

        # with random weights
        model.eval()
        output,_ = model(inpdata)
        # output=model.compute_plda_affinity_matrix(model(inpdata))
        output = output.cpu().detach().numpy()[0]
        per_loss = opt.eta
                  
    
        print('validation triplets:',len(tripletsval))
        out_val,_ = model(inpdata)
        val_loss = self.compute_loss(out_val[0],tripletsval.reshape(-1,4),alpha)
        print('validation_loss_initial:',val_loss)
        decay = 0.1
        for epoch in range(opt.epochs):
            tot_avg_loss = 0
            decay = 0.1
    
            # shuffle minibatches everytime
            random.shuffle(minibatches)
            for m,minibatch in enumerate(minibatches):
                # print('current lr:',current_lr)
                model.train()
                model.zero_grad()
                self.optimizer.zero_grad()
                
                out_train,_ = model(inpdata)
                triplet_avg_loss = self.compute_loss(out_train[0],minibatch,alpha)
               
                tot_avg_loss = triplet_avg_loss
                 
                if m==0:
                    if epoch == 0:
                        previous_val_loss = val_loss
                    else:
                        previous_val_loss = tot_val_loss
                else:
                    previous_val_loss = minibatch_val_loss
                minibatch_val_loss = self.compute_loss(out_train[0],tripletsval.reshape(-1,4),alpha)
                diff_val_loss = previous_val_loss - minibatch_val_loss
                print('\n[minibatch %d] current_lr: %f minibatch_val_loss: %.3f' % (m+1,current_lr,minibatch_val_loss))
                if m>0 and diff_val_loss > decay:
                    current_lr = current_lr/10
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                # if (m+1)%2==0:
                decay = decay/2
                if m>0 and current_lr <= 1e-6:
                    print('\n[minibatch %d] current_lr: %f decay: %.5f  minibatch_val_loss: %.3f' % (m+1,current_lr,decay,minibatch_val_loss))
                    current_lr = 1e-3/10**(epoch+1) # reset 
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                    break
    
                triplet_avg_loss.backward()
                self.optimizer.step()
    
            out_val,_ = model(inpdata)
            tot_val_loss = self.compute_loss(out_val[0],tripletsval.reshape(-1,4),alpha)
    
            print("\n[epoch %d]  triplet_val_loss: %.5f " % (epoch+1,tot_val_loss))
            if tot_val_loss < per_loss*val_loss or current_lr <= 1e-6 :
                break
        period = period + 1

        print('DER before training with n_clusters:',n_prime)
        _,_=self.validate_path_integral(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        count +=1
        self.final = 1
        model.eval()
        output1,_ = model(inpdata)
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        print('System DER with n_clusters:',n_prime)
        valcluster,val_label=self.validate_path_integral(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        count +=1

        print('Saving learnt parameters')
        matrixfold = "%s/cosine_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        # save in pickle
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
              pickle.dump(savedict,sf)

        # save in matfile
        # matrixfile = matrixfold + '/'+f+'.mat'
        # sio.savemat(matrixfile, {'output': output})

        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')
        
        return model

    def train_with_amiPIC(self,model_init,pretrain=0):
        """
        Train the SSC using N* clusters using PIC

        Parameters
        ----------
       Initial model

        Returns
        -------
        None.

        """
        model = self.model
        optimizer = self.optimizer
        alpha = 1.0 # final stage
        count = 0
        f = self.fname
        data = self.data
        
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        count +=1
        inpdata =  data.float().to(device)
        nframe = data.size()[1]

        n_prime = self.n_prime
        print('starting cluster: ',nframe)

     
        max_spks = n_prime
        period0len = n_prime

        current_lr = opt.lr
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)

        model.eval()
        # initialize filewise PCA using svd                  
        if not pretrain:
            PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
            model.init_weights(PCA_transform)
            output_model,_ = model(inpdata)
            # output_model = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
            
            cosinefold = 'cosine_pca_baseline/{}_scores/cosine_scores/'.format(dataset)
            # # cosinefold = 'scores_plda_new/{}_scores/'.format(dataset,f)
            cosinefile = '{}/{}.npy'.format(cosinefold,f)
            if not os.path.isdir(cosinefold):
                os.makedirs(cosinefold)
           
            if not os.path.isfile(cosinefile):
                np.save(cosinefile,output_model) 
        else:
            model = model_init
            output_model,_ = model(inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
        labelfull_feed=np.arange(output_model.shape[0])
        clusterlen_feed=[1]*len(labelfull_feed)    
       

        clusterlen_old = clusterlen_feed.copy()
        labelfull_old = labelfull_feed.copy()
        n_clusters = max(period0len,n_prime)
   
        distance_matrix = (output_model+1)/2
        mypic =PIC_org(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),K=self.K,z=self.z) 
        labelfull,clusterlen = mypic.gacCluster() 
                     
        cluster = self.compute_cluster(labelfull)

        minibatches,batchsize,tripletsval = self.compute_minibatches_train_valid(output_model1, cluster, labelfull,clusterlen=clusterlen)

        # with random weights
        model.eval()
        output,_ = model(inpdata)
        # output=model.compute_plda_affinity_matrix(model(inpdata))
        output = output.cpu().detach().numpy()[0]
        per_loss = opt.eta
                  
    
        print('validation triplets:',len(tripletsval))
        out_val,_ = model(inpdata)
        val_loss = self.compute_loss(out_val[0],tripletsval.reshape(-1,4),alpha)
        print('validation_loss_initial:',val_loss)
        decay = 0.1
        for epoch in range(opt.epochs):
            tot_avg_loss = 0
            decay = 0.1
    
            # shuffle minibatches everytime
            random.shuffle(minibatches)
            for m,minibatch in enumerate(minibatches):
                # print('current lr:',current_lr)
                model.train()
                model.zero_grad()
                self.optimizer.zero_grad()
                
                out_train,_ = model(inpdata)
                triplet_avg_loss = self.compute_loss(out_train[0],minibatch,alpha)
               
                tot_avg_loss = triplet_avg_loss
                 
                if m==0:
                    if epoch == 0:
                        previous_val_loss = val_loss
                    else:
                        previous_val_loss = tot_val_loss
                else:
                    previous_val_loss = minibatch_val_loss
                minibatch_val_loss = self.compute_loss(out_train[0],tripletsval.reshape(-1,4),alpha)
                diff_val_loss = previous_val_loss - minibatch_val_loss
                print('\n[minibatch %d] current_lr: %f minibatch_val_loss: %.3f' % (m+1,current_lr,minibatch_val_loss))
                if m>0 and diff_val_loss > decay:
                    current_lr = current_lr/10
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                # if (m+1)%2==0:
                decay = decay/2
                if m>0 and current_lr <= 1e-6:
                    print('\n[minibatch %d] current_lr: %f decay: %.5f  minibatch_val_loss: %.3f' % (m+1,current_lr,decay,minibatch_val_loss))
                    current_lr = 1e-3/10**(epoch+1) # reset 
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                    break
    
                triplet_avg_loss.backward()
                self.optimizer.step()
    
            out_val,_ = model(inpdata)
            tot_val_loss = self.compute_loss(out_val[0],tripletsval.reshape(-1,4),alpha)
    
            print("\n[epoch %d]  triplet_val_loss: %.5f " % (epoch+1,tot_val_loss))
            if tot_val_loss < per_loss*val_loss or current_lr <= 1e-6 :
                break


        print('DER before training with n_clusters:',n_prime)
        _,_=self.validate_path_integral(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        count +=1
        self.final = 1
        model.eval()
        output1,_ = model(inpdata)
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        print('System DER with n_clusters:',n_prime)
        valcluster,val_label=self.validate_path_integral(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        count +=1

        print('Saving learnt parameters')
        matrixfold = "%s/cosine_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        # save in pickle
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
              pickle.dump(savedict,sf)

        # save in matfile
        # matrixfile = matrixfold + '/'+f+'.mat'
        # sio.savemat(matrixfile, {'output': output})

        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')


    def train_with_callhomePIC(self,model_init,pretrain=0):
        """
        train the network using pretrained autoencoder
        case1: start training from 10 clusters more than n_prime to use x-vectors directly and then merge clusters
              in small steps to allow the network to learn. Stop at n_prime
        case2: start at 20 clusters. Go to till max_spks (10 here) in steps for learning. Then use threshold/n_prime to get final output

        **Replace max_spks with n_prime and vice versa to switch from case2 and case1 and otherwise


        Parameters
        ----------
        auto_encoder_loss_fn :
            MSE Loss

        Returns
        -------
        None.

        """

        model = self.model
        optimizer = self.optimizer
        alpha = 1.0 # final stage 
        count = 0
        f = self.fname
        data = self.data
        
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        inpdata =  data.float().to(device)
        nframe = data.size()[1]
       

        n_prime = self.n_prime
        print('starting cluster: ',nframe)
         
        max_spks = n_prime
        period0len = n_prime
      
        current_lr = opt.lr

        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)

        model.eval()
        if not pretrain:
            PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
            model.init_weights(PCA_transform)
            output_model,_ = model(inpdata)
            # output_model = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
            
            cosinefold = 'cosine_pca_baseline/{}_scores/cosine_pca30_scores/'.format(dataset)
            cosinefile = '{}/{}.npy'.format(cosinefold,f)
            if not os.path.isdir(cosinefold):
                os.makedirs(cosinefold)
            if not os.path.isfile(cosinefile):
                np.save(cosinefile,output_model) 
        else:
            model = model_init
            output_model,_ = model(inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
        labelfull_feed=np.arange(output_model.shape[0])
        clusterlen_feed=[1]*len(labelfull_feed)    

        clusterlen_old = clusterlen_feed.copy()
        labelfull_old = labelfull_feed.copy()
        n_clusters = max(period0len,n_prime)
 
        distance_matrix = output_model+1
        distance_matrix = distance_matrix/2
        if opt.threshold == None:
            mypic =PIC_callhome(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),K=self.K,z=self.z) 
        else:
            mypic =PIC_callhome_threshold(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),opt.threshold,K=self.K,z=self.z) 

        labelfull,clusterlen = mypic.gacCluster() 
        n_clusters = len(clusterlen)
        cluster = self.compute_cluster(labelfull)
        count +=1
        labelfull_xvec = labelfull.copy()
        clusterlen_xvec = clusterlen.copy()
        if n_clusters == 1:
           alpha = 0.0

        minibatches = self.compute_minibatches_train(period,output_model1, cluster, labelfull,clusterlen=clusterlen)
       
        # with random weights
        model.eval()
        output,_ = model(inpdata)
        # output=model.compute_plda_affinity_matrix(model(inpdata))
        output = output.cpu().detach().numpy()[0]

        per_loss = opt.eta

        for epoch in range(opt.epochs):
            
            model.train()

            model.zero_grad()
            self.optimizer.zero_grad()

            out_train,_ = model(inpdata)
            triplet_avg_loss = self.compute_loss(out_train[0],minibatches.reshape(-1,4),alpha)
            # tot_avg_loss = self.alpha*auto_encoder_avg_loss + self.beta * triplet_avg_loss
            tot_avg_loss = triplet_avg_loss
            print("\n[epoch %d]  triplet_avg_loss: %.5f " % (epoch+1,triplet_avg_loss))
            if epoch == 0:
                avg_loss = tot_avg_loss
            if tot_avg_loss < per_loss*avg_loss:
                break
            tot_avg_loss.backward()
            self.optimizer.step()

        print(' Pre-training DER with n_clusters:',n_prime)
        _,_=self.validate_path_integral(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        count +=1
        self.final = 1
        model.eval()
        output1,_ = model(inpdata)
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        print('SSC-PIC DER with n_clusters:',n_prime)
        valcluster,val_label=self.validate_path_integral(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        count +=1
        print('Saving learnt parameters')
        matrixfold = "%s/cosine_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output

        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        # save in pickle
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
             pickle.dump(savedict,sf)

        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')
                
    def validate_path_integral(self,output_new, period,n_clusters,clusterlen,labelfull,flag):
            f = self.fname
            overlap =0
            clusterlen_org = clusterlen.copy()
            distance_matrix = (output_new+1)/2
            if "callhome" in dataset: 
                mypic =PIC_callhome(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),K=self.K,z=self.z) 
            else:
                if opt.threshold == None:
                    mypic =PIC_ami(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),K=self.K,z=self.z) 
                else:
                    mypic =PIC_ami_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),opt.threshold,K=self.K,z=self.z) 

            labelfull,clusterlen = mypic.gacCluster()

            print('clusterlen:',clusterlen)
            self.results_dict[f]=labelfull
            if self.final:
                if self.forcing_label:
                    out_file=opt.outf+'/'+'final_rttms_forced_labels/'
                else:
                    out_file=opt.outf+'/'+'final_rttms/'
            else:
                out_file=opt.outf+'/'+'rttms/'
            if not os.path.isdir(out_file):
                os.makedirs(out_file)
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            rttm_gndfile = 'rttm_'+dataset+'_ground/'+f+'.rttm'
            self.write_results_dict(out_file)
            # bp()
            der=self.compute_score(rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                der = self.compute_score(rttm_gndfile,rttm_newfile,outpath,overlap)
            print("\n%s [period %d] DER: %.2f" % (self.fname,period, der))
            cluster = self.compute_cluster(labelfull)

            return cluster,labelfull
    def validate(self,output_new, period,n_clusters,clusterlen,labelfull,flag):
            # lamda = 0
            f = self.fname
            overlap =0
            clusterlen_org = clusterlen.copy()
            if opt.threshold == 'None' or self.final==0:
                myahc =ahc.clustering(n_clusters, clusterlen_org,self.lamda,labelfull,dist=None)
            else:
                myahc =ahc.clustering(None, clusterlen_org,self.lamda,labelfull,dist=float(opt.threshold))
            labelfull,clusterlen,_ = myahc.my_clustering_full(output_new)
            print('clusterlen:',clusterlen)
            self.results_dict[f]=labelfull
            if self.final:
                if self.forcing_label:
                    out_file=opt.outf+'/'+'final_rttms_forced_labels/'
                else:
                    out_file=opt.outf+'/'+'final_rttms/'
            else:
                out_file=opt.outf+'/'+'rttms/'
            if not os.path.isdir(out_file):
                os.makedirs(out_file)
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            rttm_gndfile = 'rttm_ground/'+f+'.rttm'
            self.write_results_dict(out_file)
            # bp()
            der=self.compute_score(rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = self.compute_score(rttm_gndfile,rttm_newfile,outpath,overlap)
            
            print("\n%s [period %d] DER: %.2f" % (self.fname,period, der))
           
            cluster = self.compute_cluster(labelfull)

            return cluster,labelfull


# hyper-parameters

keep_prob = 1 # 0.7
seed2=999
random.seed(seed2)
def main():
    seed=555
    if "Callhome" in dataset:
        xvecD=128
        pca_dim = 10
    elif "AMI" in dataset:
        xvecD=512
        pca_dim=30
    pair_list = open(opt.reco2utt_list).readlines()
    filelen =len(pair_list)
    if opt.reco2num_spk !="None":
        reco2num = open(opt.reco2num_spk).readlines()
    else:
        reco2num = "None"
    
    
    kaldimodel = pickle.load(open(opt.kaldimodel,'rb')) # PCA Transform and mean of heldout set
    ind = list(np.arange(filelen))
    random.shuffle(ind)    
   
    
    for i in range(filelen):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('pca_dim:',pca_dim)
        net_init = weight_initialization(kaldimodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        net = Deep_Ahc_model(kaldimodel,dimension=xvecD,red_dimension=pca_dim,device=device)
        model = net.to(device)

        # Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        
        # training
        reco2utt = pair_list[i]
        if reco2num != "None":
            n_prime = int(reco2num[i].split()[1])
        else:
            n_prime = 2 # needs atleast 2 clusters
        fname = reco2num[i].split()[0]         
      
        print('output_folder:',opt.outf)
        ahc_obj = Deep_AHC(kaldimodel,fname,reco2utt,xvecD,model,optimizer,n_prime)
        ahc_obj.dataloader_from_list()
        if "callhome" in opt.dataset:             
            model = ahc_obj.train_with_AHCthreshold(model_init)
            ahc_obj.train_with_callhomePIC(model,pretrain=1)
        else:
            if reco2num!="None":
                ahc_obj.train_with_amiPIC(model_init,pretrain=0)
            else:
                ahc_obj.train_with_amiPIC_threshold(model_init,pretrain=0)
        print('output_folder:',opt.outf)


if __name__ == "__main__":
    main()

