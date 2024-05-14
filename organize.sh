exp_dir=$1
src_lan=$2
tgt_lan=$3

mkdir $exp_dir
cd $exp_dir

wget https://indictrans2-public.objectstore.e2enetworks.net/indic-indic-spm.zip
wget https://indictrans2-public.objectstore.e2enetworks.net/indic-indic-fairseq-dict.zip
unzip indic-indic-spm.zip
unzip indic-indic-fairseq-dict.zip
mv indic-indic-spm vocab
mv indic-indic-fairseq-dict final_bin
rm -rf indic-indic-spm.zip
rm -rf indic-indic-fairseq-dict.zip

mkdir train
cd train
mkdir ${src_lan}-${tgt_lan}
cd ..

mkdir devtest
cd devtest
mkdir all
cd all
mkdir ${src_lan}-${tgt_lan}
