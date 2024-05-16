exp_dir=$1
src_lan=$2
tgt_lan=$3

mkdir $exp_dir
cd $exp_dir

wget https://indictrans2-public.objectstore.e2enetworks.net/indic-en-spm.zip
wget https://indictrans2-public.objectstore.e2enetworks.net/indic-en-fairseq-dict.zip
unzip indic-en-spm.zip
unzip indic-en-fairseq-dict.zip
mv indic-en-spm vocab
mv indic-en-fairseq-dict final_bin
rm -rf indic-en-spm.zip
rm -rf indic-en-fairseq-dict.zip

mkdir train
cd train
mkdir ${src_lan}-${tgt_lan}
cd ..

mkdir devtest
cd devtest
mkdir all
cd all
mkdir ${src_lan}-${tgt_lan}
