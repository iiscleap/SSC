
if [ $# != 2 ];then
    echo "Usage:$0 <kaldi_path> <ami_corpus_dir>"
    echo "kaldi_path: Path to kaldi root  directory"
    echo "ami_corpus_dir: Path for AMI corpus directory."
    exit 1
fi

kaldi_path=$1
amicorpus=$2

if [ ! -d $amicorpus ];then
    echo "$0:ERROR: $amicorpus directory does not exist."
    exit 1
fi


if [ ! -d $kaldi_path ];then
    echo "$0:ERROR: KALDI ROOT folder not present at $kaldi_path."
    exit 1
fi
ln -sf $kaldi_path .

AMI_Setup="AMI-diarization-setup"

if [ ! -d $AMI_Setup ];then
    echo "$0:ERROR:${AMI_Setup} not present. Kindly place setup in current directory or create soft link to setup."
    echo "$0:eg: ln -sf /data1/ami_dataset/AMI-diarization-setup ."
    exit 1
fi

util_path=$kaldi_path/egs/wsj/s5/utils
if [ ! -d $util_path ];then
    echo "$0:ERROR:utils folder not present at $util_path. Problrm in kaldi setup."
    exit 1
fi
ln -sf $util_path .

step_path=$kaldi_path/egs/wsj/s5/steps
if [ ! -d $step_path ];then
    echo "$0:ERROR:steps folder not present at $step_path. Problrm in kaldi setup."
    exit 1
fi
ln -sf $step_path .


for dataset in sdm
do
    for data_type in ami_eval ami_dev
    do
        ./create_data_dir.sh $amicorpus data/${data_type} $dataset $data_type
    done
done

