cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.5
rttm_channel=0
lamda=0.1
read_costs=false
reco2num_spk=
reco2utt=exp/${dataset}/spk2utt
score_path=../scores_plda_new/${dataset}_scores/
score_file=my_file
dataset=callhome1
clustering=PIC
k=30
z=0.1
which_python=/home/prachis/miniconda3/envs/mytorch/bin/python
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

echo $reco2num_spk


srcdir=$1
dir=$2

mkdir -p $dir/log
mkdir -p $dir/tmp

# echo $srcdir
# echo $score_file

cp $srcdir/spk2utt $dir/tmp/spk2utt
cp $srcdir/utt2spk $dir/tmp/utt2spk
cp $score_file $dir/tmp/dataset.list
cp $srcdir/segments $dir/tmp/segments

# Checks if the given string operand size is not zero; if it is not zero length, then it returns true.
if [ ! -z $reco2num_spk ]; then
  cp $reco2num_spk $dir/tmp/reco2num_spk
fi

utils/fix_data_dir.sh $dir/tmp > /dev/null
# echo "fix"

sdata=$dir/tmp/split$nj;
utils/split_data_mine.sh $dir/tmp $nj || exit 1;


# echo "split"
JOB=1
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  echo "log at $dir/log/agglomerative_cluster.*.log"
  if [ ! -z $reco2num_spk ]; then
    $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
      $which_python ../services/agglomerative.py --reco2num $sdata/JOB/reco2num_spk  --reco2utt $sdata/JOB/spk2utt \
      --score_path $score_path \
      --label-out $dir/labels.JOB \
      --dataset $dataset \
      --score_file $sdata/JOB/dataset.list \
      --k $k \
      --z $z \
      --clustering $clustering

  else
    echo "threshold: "$threshold
    $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
      $which_python ../services/agglomerative.py --threshold $threshold  --reco2utt $sdata/JOB/spk2utt \
      --score_path $score_path \
      --label-out $dir/labels.JOB \
      --score_file $sdata/JOB/dataset.list \
      --dataset $dataset \
      --k $k \
      --z $z \
      --clustering $clustering

  fi

fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  cat $dir/labels.* > $dir/labels || exit 1;
  # for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
  # for j in $(seq $nj); do $which_python diarization/make_rttm.py --rttm-channel $rttm_channel $sdata/$j/segments $dir/labels.$j $dir/rttm.$j; done

fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  $which_python diarization/make_rttm.py --rttm-channel $rttm_channel $srcdir/segments $dir/labels $dir/rttm || exit 1;
  
fi

rm -r $dir/tmp
