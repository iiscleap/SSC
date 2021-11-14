# specified folder will be created here in kaldi format for AMI Data sel
#  reco2dur  rttm  segments  spk2utt  utt2spk  wav.scp

if [ $# != 4 ];then
    echo "$0:ERROR: Incorrect usage."
    echo "Usage:$0 <ami_corpus_dir> <data_dir_path> <dataset> <data_type>"
    echo "ami_corpus_dir: Path to AMI corpus directory."
    echo "data_dir_path: Path for data folder to be created."
    echo "dataset: Type of dataset sdm/mdm/dev"
    echo "data_type: Type of data ami_dev/ami_eval"
    echo "sdm: Single Distant Microphone"
    
    exit 1
fi

amicorpus=$1
data_dir=$2
dataset=$3
data_type=$4

if [ ! -d $amicorpus ];then
    echo "$0:ERROR: AMI data directory not present at path: $amicorpus"
    exit 1
fi

if [ $dataset != "sdm" ];then
    echo "$0:ERROR: Incorrect dataset type. Must be sdm"
    exit 1
fi

if  [ $data_type != "ami_eval" ] && [ $data_type != "ami_dev" ];then
    echo "$0:ERROR: Incorrect data type. Must be ami_eval or ami_dev"
    exit 1
fi

if [ ! -d $data_dir ];then
    echo "$0:INFO: $data_dir not prsent. Creating $data_dir"
    mkdir -p $data_dir
fi

AMI_Setup="ami_xvector_models/AMI-diarization-setup"

if [ ! -d $AMI_Setup ];then
    echo "$0:ERROR:${AMI_Setup} not present."
    
fi

if [ ! -d "utils" ];then
    echo "$0:ERROR:Softlink to kaldi utils folder not present. Kindly create soft link to kaldi utils folder."
    echo "$0:eg: ln -sf $KALDI_ROOT/egs/wsj/s5/utils ."
    exit 1
fi


# make rttm file
if [ ! -d "${AMI_Setup}/AMI-diarization-setup/rttms/${data_type}" ];then
    echo "$0:ERROR: Problem in AMI-diarization-setup. Path : ${AMI_Setup}/rttms/${data_type} not present."
    exit 1
fi
cat ${AMI_Setup}/rttms/${data_type}/*.rttm > ${data_dir}/rttm

cp -r ${AMI_Setup}/rttms/${data_type} ${data_dir}/filewise_rttms

awk '{print $2}' ${data_dir}/rttm | sort | uniq > ${data_dir}/${data_type}.list

# create wav.scp
python wavs.py $amicorpus $data_dir $dataset 
if [ $? == 1 ];then
    echo "$0:ERROR: wav.scp file creation process failed."
    exit 1
fi

# segments and utt2spk fils
# 1. create reco2file_and_channel
awk '{print $1" "$1" "1}' ${data_dir}/${data_type}.list > ${data_dir}/reco2file_and_channel
# 2. create utt2spk and segment file
python rttm2utt2spkseg.py \
    $data_dir/rttm $data_dir/reco2file_and_channel \
    $data_dir/utt2spk $data_dir/segments --use-reco-id-as-spkr=true

if [ $? == 1 ];then
    echo "$0:ERROR: utt2spk and segments file creation process failed."
    exit 1
fi

# create spk2utt and reco2dur
# use kaldi's utility functions

utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt

utils/data/get_reco2dur.sh $data_dir

utils/fix_data_dir.sh $data_dir

