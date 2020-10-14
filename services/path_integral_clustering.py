import numpy as np
import argparse
import sys
import os
# import matplotlib.pyplot as plt
import pickle
from pdb import set_trace as bp
import subprocess
import scipy.io as sio
import time
from scipy.sparse import coo_matrix
from numpy import linalg

class PIC_ami_threshold:
    def __init__(self,n_clusters,clusterlen,labels,affinity,threshold=0.5,K=20,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.z = z
        self.A = (-1)*affinity.copy() 
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
        self.th = threshold
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #  Compute structural complexity from the subpart of the weighted adjacency matrix
        #  Input:
        #    - subIminuszW: the subpart of (I - z*P)
        #  Output:
        # 	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #  Compute conditional complexity from the subpart of the weighted adjacency matrix
        #  Inputs:
        #    - IminuszW: the matrix (I - z*P)
        # 	- cluster_i: index vector of cluster i
        # 	- cluster_j: index vector of cluster j
        #  Output:
        # 	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        #  detecting cross elements (this check costs much and is unnecessary)
        #  if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        #  compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #  NN indices
        # N = self.A.shape[0]
       
        #  find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]


        # build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        NI = NNIndex[:, 1:self.K+1].copy()
        # XI = repmat([1:N]', 1, K);
        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
       
        # beta1 = 1;
        
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()
        graphW[np.diag_indices(self.N)]=1
        
        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is distance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        # bp()
        return intialClusters

    def gacMerging(self,graphW, initClusters):
        # Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        IminuszW = np.eye(numSample) - self.z*graphW
        # del graphW
        myInf = 1e10

        # initialization
        if self.strDescr == 'path':
            complexity_fun = PIC_org.gacPathEntropy
            conditionalComplexity_fun = PIC_org.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        # compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #  compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        clusterLabels = np.ones((numSample,),dtype=int)
        # erg = 3
        th = self.th
        # groupNumber = 1 #  default
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
                # print('Group count: %d' % curGroupNum)
                # print('minima:{}'.format(minima))
            # % Find two clusters with the best affinity
            if curGroupNum <= 10: 
                
                clusterlen = []
                for i,lab in enumerate(initClusters):
                    clusterlen.append(len(lab))
                    clusterLabels[lab] = i
               
                len_initclusters = np.array(clusterlen)
                min_len = 10
                smallest_clusters = np.where(len_initclusters<min_len)[0]
                if len(smallest_clusters)==0:
                    min_len = int(numSample/curGroupNum * 0.2)
                    smallest_clusters = np.where(len_initclusters<min_len)[0]
                iu1 = np.triu_indices(curGroupNum,k=1)
                if np.sum(affinityTab[iu1])==0:
                    break
                if curGroupNum == 10:
                    S = np.zeros((curGroupNum,curGroupNum))
                    
                    S[iu1] = affinityTab[iu1]
                    S = S + S.T
                    
                    minS = np.amin(S[iu1])
                    # minS = np.sum(S,axis=1)
                    S[np.diag_indices(curGroupNum)]  = minS
                    S = (-1)*S
                    ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
                    total_energy = np.sum(eig_s)
                    print('total_energy:{}'.format(total_energy))
                    energy =0.0
                    dim=1
                    # bp()
                    while energy/total_energy <= th:
                        energy += eig_s[dim-1]
                        dim +=1
                    groupNumber = dim
                    print('dim:',dim)
                    if curGroupNum == groupNumber:
                        break

                if curGroupNum == max(groupNumber+1,dim):                    
                    num_with_min_cardinality = len(len_initclusters[len_initclusters >= min(round(0.1*numSample),10)])
                    print('num_with_min_cardinality:',num_with_min_cardinality)
                    groupNumber = min(num_with_min_cardinality,groupNumber)
                    if curGroupNum == groupNumber:
                        break

                if len(smallest_clusters)>0:
                    

                    minAff_a = np.amin(affinityTab[smallest_clusters,:curGroupNum], axis=1).reshape(1,-1)
                    minIndex1_a = np.argmin(affinityTab[smallest_clusters,:curGroupNum], axis=1).reshape(1,-1)
                    
                    minAff_b = np.amin(affinityTab[:curGroupNum, smallest_clusters],axis=0).reshape(1,-1)
                    minIndex1_b = np.argmin(affinityTab[:curGroupNum, smallest_clusters],axis=0).reshape(1,-1)
                   
                    minAff = np.amin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = np.argmin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = min_col.T
                    minIndex = np.vstack((minIndex1_a,minIndex1_b))
                    minIndex1 = minIndex[(min_col,np.arange(len(min_col)))]
                    # minIndex1 = minIndex(sub2ind(size(minIndex),min_col,1:size(min_col,2)));
                    prev_minima = np.amin(minAff)
                    # print('minima:{}'.format(prev_minima))
                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]
                    minIndex2  = smallest_clusters[minIndex2]
                    
                else:
                   
                
                    minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                    minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                    minima = np.amin(minAff)

                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]  

                  
            else:
                
                minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                minima = np.amin(minAff)
                minIndex2 = np.argmin(minAff)
                minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1
        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen

class PIC_callhome_threshold:
    def __init__(self,n_clusters,clusterlen,labels,affinity,threshold=0.5,K=20,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.z = z
        self.A = (-1)*affinity
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
        self.th = threshold
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #   Compute structural complexity from the subpart of the weighted adjacency matrix
        # % Input:
        # %   - subIminuszW: the subpart of (I - z*P)
        # % Output:
        # %	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #   Compute conditional complexity from the subpart of the weighted adjacency matrix
        # % Inputs:
        # %   - IminuszW: the matrix (I - z*P)
        # %	- cluster_i: index vector of cluster i
        # %	- cluster_j: index vector of cluster j
        # % Output:
        # %	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        # % detecting cross elements (this check costs much and is unnecessary)
        # % if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        # % compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #   NN indices
        # N = self.A.shape[0]
       
        # % find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # % sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]

        #   build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        # ND = np.exp(ND)
        NI = NNIndex[:, 1:self.K+1].copy()
        # XI = repmat([1:N]', 1, K);
        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
       
        # beta1 = 1;
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()
        graphW[np.diag_indices(self.N)]=1
        
        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is diastance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return intialClusters

    def gacMerging(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        o1=np.eye(numSample)
        o2=self.z*graphW
        IminuszW = o1-o2
        del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = self.gacPathEntropy
            conditionalComplexity_fun = self.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        
        clusterLabels = np.ones((numSample,),dtype=int)
        # groupNumber = 1 # minimum possible speakers
        groupNumber_actual = groupNumber
        tmp = 10
        th = self.th
        
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
            if curGroupNum <= 10 : 
                

                clusterlen = []
                for i,lab in enumerate(initClusters):
                    clusterLabels[lab] = i
                    clusterlen.append(len(lab))
               
                len_initclusters = np.array(clusterlen)
                min_len = max(tmp,round(numSample/curGroupNum * 0.2))
                # tmp = tmp-1
                smallest_clusters = np.where(len_initclusters<min_len)[0]


                iu1 = np.triu_indices(curGroupNum,k=1)
                my_affinity = affinityTab

                if np.sum(affinityTab[iu1])==0:
                    break
                if curGroupNum == 10:
                    S = np.zeros((curGroupNum,curGroupNum))
                    
                    S[iu1] = affinityTab[iu1]
                    S = S + S.T
                    
                    minS = np.amin(S[iu1])
                    # minS = np.sum(S,axis=1)
                    S[np.diag_indices(curGroupNum)]  = minS
                    S = (-1)*S
                    ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
                    total_energy = np.sum(eig_s)
                    print('total_energy:{}'.format(total_energy))
                    energy =0.0
                    dim=1
                    # bp()
                    while energy/total_energy <= th:
                        energy += eig_s[dim-1]
                        dim +=1    
                    print('dim:',dim)
                    num_with_min_cardinality = len(len_initclusters[len_initclusters >= min(round(0.1*numSample),10)])
                    
                    groupNumber = dim 
                    groupNumber=max(groupNumber_actual,groupNumber)
                    if curGroupNum == groupNumber: 
                        print('num_with_min_cardinality:',num_with_min_cardinality)
                        groupNumber = min(dim,num_with_min_cardinality)    
                        groupNumber=max(groupNumber_actual,groupNumber)
                        if curGroupNum == groupNumber:                   
                            break

                if curGroupNum == max(groupNumber+1,dim):
                    
                    num_with_min_cardinality = len(len_initclusters[len_initclusters >= min(round(0.1*numSample),10)])
                    print('num_with_min_cardinality:',num_with_min_cardinality)
                    groupNumber = min(num_with_min_cardinality,groupNumber)
                    groupNumber=max(groupNumber_actual,groupNumber)
                    if curGroupNum == groupNumber:
                        break

                minAff = np.amin(my_affinity[:curGroupNum, :curGroupNum],axis= 0)
                minIndex1 = np.argmin(my_affinity[:curGroupNum, :curGroupNum],axis= 0)

                minIndex2 = np.argmin(minAff)
                minIndex1 = minIndex1[minIndex2] 
                    
                    
            else:        
                minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)

                minIndex2 = np.argmin(minAff)
                minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1
        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen

class PIC_callhome:
    def __init__(self,n_clusters,clusterlen,labels,affinity,K=20,a=1,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.a = a
        self.z = z
        self.A = (-1)*affinity
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #   Compute structural complexity from the subpart of the weighted adjacency matrix
        # % Input:
        # %   - subIminuszW: the subpart of (I - z*P)
        # % Output:
        # %	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #   Compute conditional complexity from the subpart of the weighted adjacency matrix
        # % Inputs:
        # %   - IminuszW: the matrix (I - z*P)
        # %	- cluster_i: index vector of cluster i
        # %	- cluster_j: index vector of cluster j
        # % Output:
        # %	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        # % detecting cross elements (this check costs much and is unnecessary)
        # % if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        # % compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #   NN indices
        # N = self.A.shape[0]
       
        # % find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # % sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]

        #   estimate derivation
        # sig2 = mean(mean(sortedDist(:,2:max(K+1,4)))) * a;
        #     %
        # tmpNNDist = min(sortedDist(:,2:end),[],2);
        # while any(exp(- tmpNNDist / sig2) < 1e-5) % check sig2 and magnify it if it is too small
        #     sig2 = 2*sig2;
        # end
        #     %
        # disp(['  sigma = ' num2str(sqrt(sig2))]);

        #   build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        # ND = np.exp(ND)
        NI = NNIndex[:, 1:self.K+1].copy()
        # XI = repmat([1:N]', 1, K);
        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
        # if ND.shape[1]!=XI.shape[1]:
        #     bp()
        # beta1 = 1;
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()
        # graphW = full(sparse(XI(:),NI(:),-ND(:), N, N));
        # % graphW = full(sparse(XI(:),NI(:),exp(-ND(:)*(1/sig2)), N, N));
        # % graphW = full(sparse(XI(:),NI(:),1./(1+exp(beta1*ND(:))), N, N)); % sigmoid as input affinity
        graphW[np.diag_indices(self.N)]=1
        # graphW = np.exp(graphW)
        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is diastance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)
            # clusterlist.append(ind[0])
            # avg=np.sum(output_new[ind],axis=0)
            # output_new[ind[0]]=avg
            # output_new[:,ind[0]]=avg
            # initialClusters.append(find(clusterLabels(:) == uniqueLabels(i)))
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return intialClusters
    
    def gacMerging(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        o1 = np.eye(numSample)
        o2 = self.z*graphW
        IminuszW = o1 - o2
        del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = self.gacPathEntropy
            conditionalComplexity_fun = self.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        
        clusterLabels = np.ones((numSample,),dtype=int)
        
        tmp = 10
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
            # if curGroupNum <= 10 : 
                

            #     clusterlen = []
            #     for i,lab in enumerate(initClusters):
            #         clusterLabels[lab] = i
            #         clusterlen.append(len(lab))
               
            #     len_initclusters = np.array(clusterlen)
            #     # min_len = 10
            #     # smallest_clusters = np.where(len_initclusters<min_len)[0]
            #     # if len(smallest_clusters)==0:
            #     # min_len = 0
            #     # smallest_clusters = np.where(len_initclusters<min_len)[0]
                
            #     min_len = max(tmp,round(numSample/curGroupNum * 0.25))
                
            #     smallest_clusters = np.where(len_initclusters<min_len)[0]
            #     # check transition matrix
            #     # Transmat = np.zeros((curGroupNum,curGroupNum))
            #     # for i in range(numSample-1):
            #     #     Transmat[clusterLabels[i],clusterLabels[i+1]] = Transmat[clusterLabels[i],clusterLabels[i+1]] + 1
                
            #     # Transmat = Transmat/np.sum(Transmat,1,keepdims=True)
            #     # isdiagonal_dominant = 2*np.diag(Transmat)>=np.sum(Transmat,1)
            #     # isdiagonal_dominant = isdiagonal_dominant[isdiagonal_dominant==True]
            #     # diff_dominant = 2*np.diag(Transmat)-np.sum(Transmat,1)
            #     # Tnew = (Transmat+Transmat.T)/2
            #     # Tnew = Tnew/np.max(Tnew)              
            #     iu1 = np.triu_indices(curGroupNum,k=1)               
               
            
            #     # my_affinity =  np.inf * np.ones((curGroupNum,curGroupNum))
            #     # if np.sum(np.abs(affinityTab[iu1])) > 0 and np.sum(Tnew[iu1]) > 0: 
            #     #     my_affinity[iu1] = 0.95*affinityTab[iu1]/np.sum(np.abs(affinityTab[iu1])) -0.05*Tnew[iu1]/np.sum(Tnew[iu1])
            #     # else:
            #     #     my_affinity[iu1] = 0.95*affinityTab[iu1] -0.05*Tnew[iu1]

            #     my_affinity = affinityTab
            #     minAff = np.amin(my_affinity[:curGroupNum, :curGroupNum],axis= 0)
            #     minIndex1 = np.argmin(my_affinity[:curGroupNum, :curGroupNum],axis= 0)

            #     minIndex2 = np.argmin(minAff)
            #     minIndex1 = minIndex1[minIndex2]       
            # else:        
            minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
            minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)

            minIndex2 = np.argmin(minAff)
            minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacMerging_threshold(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        o1 = np.eye(numSample)
        o2 = self.z*graphW
        IminuszW = o1 - o2
        del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = self.gacPathEntropy
            conditionalComplexity_fun = self.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        
        clusterLabels = np.ones((numSample,),dtype=int)
        
        tmp = 10
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
            if curGroupNum <= 5 : 
                

                clusterlen = []
                for i,lab in enumerate(initClusters):
                    clusterLabels[lab] = i
                    clusterlen.append(len(lab))
               
                len_initclusters = np.array(clusterlen)
                min_len = min(10,round(numSample*0.1))
            #     # smallest_clusters = np.where(len_initclusters<min_len)[0]
            #     # if len(smallest_clusters)==0:
            #     # min_len = 0
            #     # smallest_clusters = np.where(len_initclusters<min_len)[0]
                
            #     min_len = max(tmp,round(numSample/curGroupNum * 0.25))
                
                smallest_clusters = np.where(len_initclusters<min_len)[0]
                if len(smallest_clusters)>0:
                    minAff_a = np.amin(affinityTab[smallest_clusters,:curGroupNum], axis=1).reshape(1,-1)
                    minIndex1_a = np.argmin(affinityTab[smallest_clusters,:curGroupNum], axis=1).reshape(1,-1)
                    
                    minAff_b = np.amin(affinityTab[:curGroupNum, smallest_clusters],axis=0).reshape(1,-1)
                    minIndex1_b = np.argmin(affinityTab[:curGroupNum, smallest_clusters],axis=0).reshape(1,-1)
                   
                    minAff = np.amin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = np.argmin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = min_col.T
                    minIndex = np.vstack((minIndex1_a,minIndex1_b))
                    minIndex1 = minIndex[(min_col,np.arange(len(min_col)))]
                    # minIndex1 = minIndex(sub2ind(size(minIndex),min_col,1:size(min_col,2)));
                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]
                    minIndex2  = smallest_clusters[minIndex2]
                else:
                    minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                    minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
        
                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]
          
         
            else:
                minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
    
                minIndex2 = np.argmin(minAff)
                minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1
        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen

class PIC_ami:
    def __init__(self,n_clusters,clusterlen,labels,affinity,K=20,a=1,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.a = a
        self.z = z
        self.A = (-1)*affinity.copy() 
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #   Compute structural complexity from the subpart of the weighted adjacency matrix
        # % Input:
        # %   - subIminuszW: the subpart of (I - z*P)
        # % Output:
        # %	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #   Compute conditional complexity from the subpart of the weighted adjacency matrix
        # % Inputs:
        # %   - IminuszW: the matrix (I - z*P)
        # %	- cluster_i: index vector of cluster i
        # %	- cluster_j: index vector of cluster j
        # % Output:
        # %	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        # % detecting cross elements (this check costs much and is unnecessary)
        # % if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        # % compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #   NN indices
        # N = self.A.shape[0]
       
        # % find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # % sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]

        #   build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        # ND = np.exp(ND)
        NI = NNIndex[:, 1:self.K+1].copy()
        # XI = repmat([1:N]', 1, K);
        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
       
        # beta1 = 1;
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()

        graphW[np.diag_indices(self.N)]=1

        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is diastance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)
           
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return intialClusters

    def gacMerging(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        IminuszW = np.eye(numSample) - self.z*graphW
        # del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = PIC_org.gacPathEntropy
            conditionalComplexity_fun = PIC_org.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        clusterLabels = np.ones((numSample,),dtype=int)
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
            if curGroupNum <= 10: 
                
                clusterlen = []
                for i,lab in enumerate(initClusters):
                    clusterlen.append(len(lab))
                    clusterLabels[lab] = i
               
                len_initclusters = np.array(clusterlen)
                
                min_len = 10
                smallest_clusters = np.where(len_initclusters<min_len)[0]
                if len(smallest_clusters)==0:
                    min_len = int(numSample/curGroupNum * 0.2)
                    smallest_clusters = np.where(len_initclusters<min_len)[0]

                             
                if len(smallest_clusters)>0:
                    
                    minAff_a = np.amin(affinityTab[np.ix_(smallest_clusters,np.arange(curGroupNum))], axis=1)
                    minIndex1_a = np.argmin(affinityTab[np.ix_(smallest_clusters,np.arange(curGroupNum))], axis=1)
                    
                    minAff_b = np.amin(affinityTab[np.ix_(np.arange(curGroupNum), smallest_clusters)],axis=0)
                    minIndex1_b = np.argmin(affinityTab[np.ix_(np.arange(curGroupNum), smallest_clusters)],axis=0)

                    minAff = np.amin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = np.argmin(np.vstack((minAff_a,minAff_b)),axis=0)
                    min_col = min_col.T
                    minIndex = np.vstack((minIndex1_a,minIndex1_b))
                    minIndex1 = minIndex[(min_col,np.arange(len(min_col)))]
                    # minIndex1 = minIndex(sub2ind(size(minIndex),min_col,1:size(min_col,2)));
                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]
                    minIndex2  = smallest_clusters[minIndex2]
                else:
                    # [minAff, minIndex1] = np.min(affinityTab[np.ix_(:curGroupNum,:curGroupNum)], axis=0)
                    # use eigengap to find optimal clusters
                    # print('curGroupNum:{}'.format(curGroupNum))
                    minAff = np.amin(affinityTab[:curGroupNum,:curGroupNum], axis=0)
                    minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                    minIndex2 = np.argmin(minAff)
                    minIndex1 = minIndex1[minIndex2]
                  
            else:
                
                minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
                minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)

                minIndex2 = np.argmin(minAff)
                minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1
        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen

class PIC_org:
    def __init__(self,n_clusters,clusterlen,labels,affinity,K=20,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.z = z
        self.A = (-1)*affinity.copy() 
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #   Compute structural complexity from the subpart of the weighted adjacency matrix
        # % Input:
        # %   - subIminuszW: the subpart of (I - z*P)
        # % Output:
        # %	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #   Compute conditional complexity from the subpart of the weighted adjacency matrix
        # % Inputs:
        # %   - IminuszW: the matrix (I - z*P)
        # %	- cluster_i: index vector of cluster i
        # %	- cluster_j: index vector of cluster j
        # % Output:
        # %	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        # % detecting cross elements (this check costs much and is unnecessary)
        # % if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        # % compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #   NN indices
        # N = self.A.shape[0]
       
        # % find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # % sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]

      
        #   build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        NI = NNIndex[:, 1:self.K+1].copy()

        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
       
        # beta1 = 1;
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()
      
        # % graphW = full(sparse(XI(:),NI(:),1./(1+exp(beta1*ND(:))), N, N)); % sigmoid as input affinity
        graphW[np.diag_indices(self.N)]=1
        
        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is diastance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)          
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return intialClusters

    def gacMerging(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        IminuszW = np.eye(numSample) - self.z*graphW
        # del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = PIC_org.gacPathEntropy
            conditionalComplexity_fun = PIC_org.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
        clusterLabels = np.ones((numSample,),dtype=int)
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
                       
                
            minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
            minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)

            minIndex2 = np.argmin(minAff)
            minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1
        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen

class PIC_org_threshold:
    def __init__(self,n_clusters,clusterlen,labels,affinity,threshold,K=20,z=0.01):
        self.n_clusters = n_clusters
        self.K = K
        self.z = z
        self.A = (-1)*affinity.copy() 
        self.strDescr = 'path'
        self.N = self.A.shape[0]
        self.clusterlen = clusterlen.copy()
        self.labelfull = labels.copy()
        self.th = threshold
    @staticmethod
    def gacPathEntropy(subIminuszW):
        #   Compute structural complexity from the subpart of the weighted adjacency matrix
        # % Input:
        # %   - subIminuszW: the subpart of (I - z*P)
        # % Output:
        # %	- clusterComp - strucutral complexity of a cluster.
        
        
        N = subIminuszW.shape[0]
        # clusterComp = np.dot(linalg.inv(subIminuszW),np.ones((N,1)))
        clusterComp = linalg.solve(subIminuszW,np.ones((N,1)))
        # clusterComp = subIminuszW \ ones(N,1);
        clusterComp = np.sum(clusterComp) / (N*N)

        return clusterComp

    @staticmethod
    def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
        #   Compute conditional complexity from the subpart of the weighted adjacency matrix
        # % Inputs:
        # %   - IminuszW: the matrix (I - z*P)
        # %	- cluster_i: index vector of cluster i
        # %	- cluster_j: index vector of cluster j
        # % Output:
        # %	- L_ij - the sum of conditional complexities of cluster i and j after merging.
        

        num_i = len(cluster_i)
        num_j = len(cluster_j)

        # % detecting cross elements (this check costs much and is unnecessary)
        # % if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

        ijGroupIndex =  cluster_i.copy()
        ijGroupIndex.extend(cluster_j)
        y_ij = np.zeros((num_i+num_j,2))  #% [y_i, y_j]
        y_ij[:num_i,0] = 1
        y_ij[num_i:,1] = 1
        
        # % compute the coditional complexity of cluster i and j
        # L_ij = np.dot(linalg.inv(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)]),y_ij)
        L_ij = linalg.solve(IminuszW[np.ix_(ijGroupIndex, ijGroupIndex)],y_ij)
        L_ij = np.sum(L_ij[:num_i,0]) / (num_i*num_i) + np.sum(L_ij[num_i:,1]) / (num_j*num_j)

        return L_ij


    def gacBuildDigraph(self):

        """
        Input:
        - A: pairwise distance, d_{i -> j}
        - K: the number of nearest neighbors for KNN graph
        - a: for covariance estimation
            sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
        - graphW: asymmetric weighted adjacency matrix, 
                    w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
                    w_{ij} = - d_{ij}
            - NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
        """
        #   NN indices
        # N = self.A.shape[0]
       
        # % find 2*K NNs in the sense of given distances
        sortedDist = np.sort(self.A,axis=1)
        NNIndex = np.argsort(self.A,axis=1)
        # % sortedDist = sortedDist.^2; % taking elementwise distance square , only if -cos as dist and W=gaussian , otherwise not needed
        NNIndex = NNIndex[:,:self.K+1]

        #   build graph
        ND = sortedDist[:, 1:self.K+1].copy()
        # ND = np.exp(ND)
        NI = NNIndex[:, 1:self.K+1].copy()
        # XI = repmat([1:N]', 1, K);
        XI = np.dot(np.arange(self.N).reshape(-1,1),np.ones((1,self.K),dtype=int))
       
        # beta1 = 1;
        graphW = coo_matrix((ND.reshape(-1), (XI.reshape(-1),  NI.reshape(-1))), shape=(self.N, self.N)).toarray()
        # % graphW = full(sparse(XI(:),NI(:),1./(1+exp(beta1*ND(:))), N, N)); % sigmoid as input affinity
        graphW[np.diag_indices(self.N)]=1
        # graphW = np.exp(graphW)
        return graphW, NNIndex

    def gacNNMerge(self,NNIndex):
        # self.A is diastance matrix here
        sampleNum = self.N
        clusterLabels = (-1)*np.ones((sampleNum, 1),dtype=int)
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            
            assignedCluster = clusterLabels[idx]
            assignedCluster_u = np.unique(assignedCluster[assignedCluster >= 0])
            if len(assignedCluster_u) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster_u) == 1:
                clusterLabels[idx] = assignedCluster_u[0]
            else:
                clusterLabels[idx] = assignedCluster_u[0];            
                for j in range(1,len(assignedCluster_u)):
                    clusterLabels[clusterLabels == assignedCluster_u[j]] = assignedCluster_u[0]
        
        uniqueLabels = np.unique(clusterLabels)
        # clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0]
        clusterlen = []
        # output_new = A.copy()
        # clusterlist=[]
        intialClusters = []
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            clusterlen.append(cluster_count)
           
            intialClusters.append((ind).tolist())
        self.clusterlen = clusterlen
        # output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return intialClusters

    def gacMerging(self,graphW, initClusters):
        # % Cluster merging for Graph Agglomerative Clustering 
        # Implements an agglomerative clustering algorithm based on maiximum graph
        # strcutural affinity of two groups
        # Inputs:
        #     - graphW: asymmetric weighted adjacency matrix
        # - initClusters: a cell array of clustered vertices
        # - groupNumber: the final number of clusters
        # - strDescr: structural descriptor, 'zeta' or 'path'
        # - z: (I - z*P), default: 0.01
        # Outputs:
        # - clusterLabels: 1 x m list whose i-th entry is the group assignment of
        #                 the i-th data vector w_i. Groups are indexed
        #                 sequentially, starting from 1. 
        VERBOSE = True
        numSample = self.N
        groupNumber = self.n_clusters
        IminuszW = np.eye(numSample) - self.z*graphW
        # del graphW
        myInf = 1e10

        #   initialization
        if self.strDescr == 'path':
            complexity_fun = PIC_org.gacPathEntropy
            conditionalComplexity_fun = PIC_org.gacPathCondEntropy
                    
        
        numClusters = len(initClusters)
        if numClusters <= groupNumber:
            ValueError('GAC: too few initial clusters. Do not need merging!')

        #   compute the structural complexity of each initial cluster
        clusterComp = np.zeros((numClusters,1))
        for i in range(numClusters):
            clusterComp[i] = complexity_fun(IminuszW[np.ix_(initClusters[i], initClusters[i])])
        
        
        #   compute initial (negative) affinity table (upper trianglar matrix), very slow
        if VERBOSE:
            print('   Computing initial table.' )
        
        affinityTab = np.inf * np.ones((numClusters,numClusters))
        for j in range(numClusters):
            for i in range(j):
                affinityTab[i, j] = - conditionalComplexity_fun(IminuszW, initClusters[i], initClusters[j])
        
        
        # affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

        affinityTab = clusterComp + clusterComp.T + affinityTab
        if VERBOSE:
            print('   Starting merging process')

        curGroupNum = numClusters
       
        th = self.th
        groupNumber_actual = groupNumber
        
        clusterLabels = np.ones((numSample,),dtype=int)
        while True: 
            # if (curGroupNum % 20 ) == 0 and VERBOSE:
            #     print('Group count: %d' % curGroupNum)
            
            # % Find two clusters with the best affinity
            if curGroupNum <= 10: 
                
                clusterlen = []
                for i,lab in enumerate(initClusters):
                    clusterlen.append(len(lab))
                    clusterLabels[lab] = i
               
                len_initclusters = np.array(clusterlen)  
                iu1 = np.triu_indices(curGroupNum,k=1)
                if np.sum(affinityTab[iu1])==0:
                    break
                if curGroupNum == 10:
                    S = np.zeros((curGroupNum,curGroupNum))
                    
                    S[iu1] = affinityTab[iu1]
                    S = S + S.T
                    
                    minS = np.amin(S[iu1])
                    # minS = np.sum(S,axis=1)
                    S[np.diag_indices(curGroupNum)]  = minS
                    S = (-1)*S
                    ev_s, eig_s , _ = np.linalg.svd(S,full_matrices=True)
                    total_energy = np.sum(eig_s)
                    print('total_energy:{}'.format(total_energy))
                    energy =0.0
                    dim=1
                    # bp()
                    while energy/total_energy <= th:
                        energy += eig_s[dim-1]
                        dim +=1
                    groupNumber = dim
                    groupNumber=max(groupNumber_actual,groupNumber)
                    print('dim:',dim)
                    if curGroupNum == groupNumber:
                        break
                    
                    
                if curGroupNum == max(groupNumber+1,dim):                    
                    num_with_min_cardinality = len(len_initclusters[len_initclusters >= min(round(0.1*numSample),10)])
                    print('num_with_min_cardinality:',num_with_min_cardinality)
                    groupNumber = min(num_with_min_cardinality,groupNumber)
                    groupNumber=max(groupNumber_actual,groupNumber)
                    if curGroupNum == groupNumber:
                        break
             

            minAff = np.amin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)
            minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum],axis= 0)

            minIndex2 = np.argmin(minAff)
            minIndex1 = minIndex1[minIndex2]            
        
            if minIndex2 < minIndex1: # swap to keep row< col
                x = minIndex1 
                minIndex1 = minIndex2
                minIndex2 = x
                

            # % merge the two clusters
            new_cluster = initClusters[minIndex1].copy()
            new_cluster.extend(initClusters[minIndex2])
            new_cluster = list(set(new_cluster))
            # new_cluster = np.unique(initClusters[minIndex1].extend(initClusters[minIndex2]))
            # % move the second cluster to be merged to the end of the cluster array
            # % note that we only need to copy the end cluster's information to
            # % the second cluster's position
            if (minIndex2 != curGroupNum-1):
                initClusters[minIndex2] = initClusters[-1]
                clusterComp[minIndex2] = clusterComp[curGroupNum-1]
                # % affinityTab is an upper triangular matrix
                affinityTab[:minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum-1]
                affinityTab[minIndex2, minIndex2+1:curGroupNum-1] = affinityTab[minIndex2+1:curGroupNum-1, curGroupNum-1]
            
            # % update the first cluster and remove the second cluster
            
            initClusters[minIndex1] = new_cluster
            initClusters.pop()
            # initClusters[-1] = []
            clusterComp[minIndex1] = complexity_fun(IminuszW[np.ix_(new_cluster, new_cluster)])
            clusterComp = clusterComp[:-1]
            affinityTab = affinityTab[:-1,:-1]
            # clusterComp[curGroupNum-1] = myInf
            # affinityTab[:,curGroupNum-1] = myInf
            # affinityTab[curGroupNum-1,:] = myInf
            curGroupNum = curGroupNum - 1
            if curGroupNum <= groupNumber:
                break
            

            # % update the affinity table for the merged cluster
            
            for groupIndex1 in  range(minIndex1):
                affinityTab[groupIndex1, minIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            for groupIndex1 in  range(minIndex1+1,curGroupNum):
                affinityTab[minIndex1, groupIndex1] = - conditionalComplexity_fun(IminuszW, initClusters[groupIndex1], new_cluster)
            
            affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1,0] + clusterComp[minIndex1,0] + affinityTab[:minIndex1, minIndex1]
            affinityTab[minIndex1, minIndex1+1:curGroupNum] = clusterComp[minIndex1+1:curGroupNum,0] + clusterComp[minIndex1,0] + affinityTab[minIndex1, minIndex1+1:curGroupNum]
        
        
        #   generate sample labels
        clusterLabels = np.ones((numSample,),dtype=int)
        clusterlen = []
        for i,lab in enumerate(initClusters):
            clusterLabels[lab] = i
            clusterlen.append(len(lab))
        self.clusterlen = clusterlen
        if VERBOSE:
            print('   Final group count: %d ' % curGroupNum)
        
        self.labelfull = clusterLabels

    def gacCluster(self):
        """
        Input: 
        - A: pairwise distances, d_{i -> j}
        - groupNumber: the final number of clusters
        - strDescr: structural descriptor. The choice can be
                        - 'zeta':  zeta function based descriptor
                        - 'path':  path integral based descriptor
        - K: the number of nearest neighbors for KNN graph, default: 20
        - p: merging (p+1)-links in l-links algorithm, default: 1

        - z: (I - z*P), default: 0.01
        Output:
        - clusteredLabels: clustering results
        """
        #   initialization

        print('---------- Building graph and forming initial clusters with l-links ---------')
        
        graphW, NNIndex = self.gacBuildDigraph()
        #% from adjacency matrix to probability transition matrix
        # graphW = bsxfun(@times, 1./sum(graphW,2), graphW); #% row sum is 1
        graphW = graphW /np.sum(graphW,axis=1).reshape(self.N,1)
        initialClusters = self.gacNNMerge(NNIndex)
        # clear distance_matrix NNIndex
        
        print('--------------------------  merging --------------------------')
        self.gacMerging(graphW, initialClusters)
        
        return self.labelfull,self.clusterlen
