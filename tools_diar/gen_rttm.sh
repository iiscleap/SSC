   
. ./cmd.sh
. ./path.sh


stage=1
DATA=Callhome
modelpath=../tf_events/xvec_ahc_folder
which_python=python # python envorionment with required libraries installed
. utils/parse_options.sh

# using oracle number of speakers 
if [ $DATA = "Callhome" ]; then
    if [ $stage -eq 1 ]; then
      # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29
        k=30
        z=0.1
        for dataset in callhome1 callhome2 ;do
          score_cosine_path=$modelpath/${dataset}_scores/
          labels=labels_prob_cos_K_${k}_z_${z}
          ./PIC_cluster.sh --cmd "$train_cmd --mem 4G" --nj 15 \
          --reco2num_spk data/$dataset/reco2num_spk --dataset ${dataset} \
         --score_path $score_cosine_path/${metric}_scores/ \
        --score_file ../lists/${dataset}/${dataset}.list --k $k --z $z \
                exp/${dataset}/ $score_cosine_path/${metric}_${labels}_oracle/ 
    
          md-eval-22.pl -r \
            data/$dataset/ref.rttm -1 -c 0.25 -s $score_cosine_path/${metric}_${labels}_oracle/rttm 2> $score_cosine_path/${metric}_${labels}_oracle/DER.log \
            > $score_cosine_path/${metric}_${labels}_oracle/DER.txt
    
            der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $score_cosine_path/${metric}_${labels}_oracle/DER.txt)
            echo "der $dataset: $der"
    
          done
          model=${metric}_${labels}_oracle
          mkdir -p $modelpath/results
        
          mkdir -p $modelpath/callhome/$model
        
          cat $modelpath/callhome1_scores/$model/rttm $modelpath/callhome2_scores/$model/rttm > $modelpath/callhome/$model/rttm
        
          cat $modelpath/callhome/$model/rttm | md-eval-22.pl -1 -c 0.25 -r \
             data/callhome/fullref.rttm -s - 2> $modelpath/results/${metric}_${labels}_oracle_threshold.log \
          > $modelpath/results/${metric}_${labels}_oracle_DER_threshold.txt
        
        
          der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $modelpath/results/${metric}_${labels}_oracle_DER_threshold.txt)
          
          echo "der callhome: $der"
        
          
          bash ../compute_rttm.sh callhome $modelpath/callhome/$model/rttm
          bash ../score.sh $modelpath/callhome/ ../ref_callhome.scp
        
          echo "filewise der at $modelpath/callhome/$model/der.scp"
        
    fi
    
    
    # thresholding PIC clustering
    
    if [ $stage -eq 2 ]; then
    

     # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29

     metric=cosine
     k=30
     z=0.1
     labels=labels_prob_cos_AHC_K_${k}_z_${z}
     mkdir -p $modelpath/tuning
     for dataset in callhome1 callhome2 ;do
        #break
        
        score_cosine_path=$modelpath/${dataset}_scores/
        threshold=0.5
        ./PIC_cluster.sh --cmd "$train_cmd" --nj 40 --threshold $threshold  --score_path $score_cosine_path/${metric}_scores/ \
         --score_file ../lists/${dataset}/${dataset}.list --k $k --z $z \
        exp/${dataset}/ $score_cosine_path/${metric}_${labels}_t$threshold/  
  
         md-eval-22.pl -1 -c 0.25 -r \
             data/$dataset/ref.rttm -s $score_cosine_path/${metric}_${labels}_t$threshold/rttm 2> $score_cosine_path/${metric}_${labels}_t$threshold/DER.log \
             > $score_cosine_path/${metric}_${labels}_t$threshold/DER.txt
  
             der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $score_cosine_path/${metric}_${labels}_t$threshold/DER.txt)
            echo "der $dataset: $der"
            
     done
  
    threshold=0.5
    model=${metric}_${labels}_t$threshold

     mkdir -p $modelpath/results
  
     mkdir -p $modelpath/callhome/$model
     
     dataset=callhome1
     dataset2=callhome2
     
     cat $modelpath/${dataset}_scores/$model/rttm $modelpath/${dataset2}_scores/$model/rttm > $modelpath/callhome/$model/rttm
   
     cat $modelpath/callhome/$model/rttm | md-eval-22.pl -1 -c 0.25 -r \
       data/callhome/fullref.rttm -s - 2> $modelpath/results/threshold_${model}.log \
       > $modelpath/results/DER_threshold_${model}.txt
  
  
     der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
      $modelpath/results/DER_threshold_${model}.txt)
     echo "der callhome: $der"
      bash ../compute_rttm.sh callhome $modelpath/callhome/$model/rttm 
      bash ../score.sh $modelpath/callhome/ ../ref_callhome.scp
        
      echo "filewise der at $modelpath/callhome/der.scp"
    
    fi
else
    if [ $stage -eq 1 ]; then
      # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29
        k=30
        z=0.1
        for dataset in ami_dev ami_eval ;do
          score_cosine_path=$modelpath/${dataset}_scores/
          labels=labels_prob_cos_K_${k}_z_${z}
          ./PIC_cluster.sh --cmd "$train_cmd --mem 4G" --nj 15 \
          --reco2num_spk data/$dataset/reco2num_spk --dataset ${dataset} \
         --score_path $score_cosine_path/${metric}_scores/ \
        --score_file ../lists/${dataset}/${dataset}.list --k $k --z $z \
                exp/${dataset}/ $score_cosine_path/${metric}_${labels}_oracle/ 
    
          md-eval-22.pl -r \
            data/$dataset/ref.rttm -1 -c 0.25 -s $score_cosine_path/${metric}_${labels}_oracle/rttm 2> $score_cosine_path/${metric}_${labels}_oracle/DER.log \
            > $score_cosine_path/${metric}_${labels}_oracle/DER.txt
    
            der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $score_cosine_path/${metric}_${labels}_oracle/DER.txt)
            echo "der $dataset: $der"
            bash ../compute_rttm.sh $dataset $modelpath/$dataset/$model/rttm 
            bash ../score.sh $modelpath/$dataset/ ../ref_${dataset}.scp
      
            echo "filewise der at $modelpath/$dataset/$model/der.scp"
          done

    fi

    # thresholding PIC clustering
    
    if [ $stage -eq 2 ]; then
     # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29
     metric=cosine
     k=30
     z=0.1
     labels=labels_prob_cos_AHC_K_${k}_z_${z}
     mkdir -p $modelpath/tuning
     for dataset in ami_dev ami_eval ;do
        score_cosine_path=$modelpath/${dataset}_scores/
        threshold=0.5
        ./PIC_cluster.sh --cmd "$train_cmd" --nj 40 --threshold $threshold  --score_path $score_cosine_path/${metric}_scores/ \
         --score_file ../lists/${dataset}/${dataset}.list --k $k --z $z \
        exp/${dataset}/ $score_cosine_path/${metric}_${labels}_t$threshold/  
  
         md-eval-22.pl -1 -c 0.25 -r \
             data/$dataset/ref.rttm -s $score_cosine_path/${metric}_${labels}_t$threshold/rttm 2> $score_cosine_path/${metric}_${labels}_t$threshold/DER.log \
             > $score_cosine_path/${metric}_${labels}_t$threshold/DER.txt
  
             der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            $score_cosine_path/${metric}_${labels}_t$threshold/DER.txt)
            echo "der $dataset: $der"
            bash ../compute_rttm.sh $dataset $modelpath/$dataset/$model/rttm 
            bash ../score.sh $modelpath/$dataset/ ../ref_${dataset}.scp
      
            echo "filewise der at $modelpath/$dataset/$model/der.scp"
            
     done

    
    
    fi

