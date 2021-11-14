#!/bin/bash

dataset=ami_dev # ami_eval for eval training
dataset2=dihard_dev_19
th=0.7
outf=model_output/xvec_ssc_trained/${dataset}_scores/
alpha=1.0
eta=0.5


nj=10
kaldi_recipe_path=/data1/prachis/SRE_19/Diarization_scores/swbd_diar/
mkdir -p $outf/

which_python=python # python environment 
. ./cmd.sh
. ./path.sh

main_dir=default
. ./utils/parse_options.sh

cosinefold=cosine_pca_baseline/${dataset}_scores/cosine_scores/
mkdir -p $cosinefold

# --reco2num_spk lists/$dataset/tmp/split$nj/JOB/reco2num_spk \

if [ $main_dir == "default" ]; then
    echo "need main_directory full path as argument"
    echo " Set arguments for training in the code"
    echo "Usage : bash run_xvec_ahc.sh --TYPE <parallel/None> --nj <number of jobs> full path of main directory --which_python <python with all requirements> "
    exit 1
fi


if [ $TYPE == "parallel" ]; then 
     if [ ! -d lists/$dataset/tmp/split$nj ] || [ ! "$(ls -A lists/$dataset/tmp/split$nj/1)" ]; then
        cd tools_diar
        utils/split_data_mine.sh $main_dir/lists/$dataset/tmp $nj || exit 1;
        cd ../
    fi
    $train_cmd JOB=1:$nj $outf/log_new/Deep_AHC.JOB.log \
    $which_python xvec_SSC_train.py \
    --which_python $which_python \
    --gpuid '1' \
    --dataset $dataset \
    --batchSize 4096 \
    --N_batches 50 \
    --epochs 10 \
    --lr 1e-3 \
    --eta $eta \
    --alpha $alpha \
    --outf $outf \
    --dataset $dataset \
    --xvecpath $kaldi_recipe_path/xvectors_npy/${dataset}/ \
    --filetrain_list lists/$dataset/tmp/split$nj/JOB/dataset.list \
    --reco2utt_list lists/$dataset/tmp/split$nj/JOB/spk2utt \
    --reco2num_spk lists/$dataset/tmp/split$nj/JOB/reco2num_spk \
    --segments lists/$dataset/segments_xvec \
    --kaldimodel lists/$dataset2/plda_${dataset2}.pkl \
    --rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/ 
else
    $which_python xvec_SSC_train.py \
    --which_python $which_python \
    --gpuid '1' \
    --dataset $dataset \
    --batchSize 4096 \
    --N_batches 50 \
    --epochs 10 \
    --lr 1e-3 \
    --eta $eta \
    --alpha $alpha \
    --outf $outf \
    --dataset $dataset \
    --xvecpath tools_diar/xvectors_npy/${dataset}/ \
    --filetrain_list lists/$dataset/tmp/dataset.list \
    --reco2utt_list lists/$dataset/tmp/spk2utt \
    --reco2num_spk lists/$dataset/tmp/reco2num_spk \
    --segments lists/$dataset/segments_xvec \
    --kaldimodel lists/ami/plda_ami.pkl \
    --rttm_ground_path $kaldi_recipe_path/data/$dataset/filewise_rttms/ 

done

