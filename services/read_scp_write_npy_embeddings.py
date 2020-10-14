import os
import numpy as np
import glob
import kaldi_io
import sys


#Usage
if len(sys.argv)!=5:
    print("Need 3 input arguments!")
    print("Usage :")
    print("python read_scp_write_npy_embeddings.py <mat/vec> <complete path of ark/scp file> <path of output folder to store numpy output> <filename list>")
    print("<mat/vec> : mat if scp contains matrix , vec if scp contains vector e.g. x-vectors")

datatype = sys.argv[1]
arkscppath = sys.argv[2] 
outputnpyfilepath = sys.argv[3] 
file_list_path = sys.argv[4]

def convert_to_npy(datatype,arkscppath,outputnpyfilepath,file_list_path):
    if not os.path.isdir(outputnpyfilepath):
    	print('Creating directory where npy scores will be saved : {}'.format(outputnpyfilepath))
    	os.makedirs(outputnpyfilepath)
    else:
        print("xvectors numpy path exists !")
        exit()
    file_name = os.path.basename(arkscppath)
    ext = os.path.splitext(file_name)[1]
    if datatype=='mat':
        #for score files
        if ext==".scp":
            d = { key:mat for key,mat in kaldi_io.read_mat_scp(arkscppath) }
        else:
            print("File type not correct. scp required.")
    elif datatype=='vec':
    	#for embeddings
        if ext==".scp":
            d = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(arkscppath) }
        elif ext == ".ark":
            d = { key:mat for key,mat in kaldi_io.read_vec_flt_ark(arkscppath) }
        else:
            print("File type not correct. scp/ark required.")
    else:
        print("first argument should be mat/vec ")
    
    
    file_list = open(file_list_path,'r').readlines()
    file_count = 0
    for count,(i,j) in enumerate(d.items()):
        if count == 0:
            system = j.reshape(1,-1)
        if count % 100 == 0:
            print("Done with {} files".format(count))
        fn = file_list[file_count].rsplit()[0]
        if fn in i:
            system = np.vstack((system,j))
        else:
            np.save(outputnpyfilepath+'/'+fn,system)
            file_count = file_count + 1
            system = j.reshape(1,-1)
        
if __name__ == "__main__":
    convert_to_npy(datatype,arkscppath,outputnpyfilepath,file_list_path)