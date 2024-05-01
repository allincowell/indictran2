#/bin/bash

root_dir=$(pwd)
echo "Setting up the environment in the $root_dir"

# --------------------------------------------------------------
#                   PyTorch Installation
# --------------------------------------------------------------
# python3.9 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118


pip3 install torch torchvision torchaudio

# --------------------------------------------------------------
#       Install IndicNLP library and necessary resources
# --------------------------------------------------------------
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
export INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources

# we use version 0.92 which is the latest in the github repo
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
cd indic_nlp_library
python3.10 -m pip install ./
cd $root_dir

# --------------------------------------------------------------
#               Install additional utility packages
# --------------------------------------------------------------
python3.10 -m pip install nltk sacremoses regex pandas mock transformers==4.28.1 sacrebleu==2.3.1 urduhack[tf] mosestokenizer ctranslate2==3.9.0 gradio
python3.10 -c "import urduhack; urduhack.download()"
python3.10 -c "import nltk; nltk.download('punkt')"

# --------------------------------------------------------------
#               Sentencepiece for tokenization
# --------------------------------------------------------------
# build the cpp binaries from the source repo in order to use the command line utility
# source repo: https://github.com/google/sentencepiece
python3.10 -m pip install sentencepiece

# --------------------------------------------------------------
#               Fairseq Installation from Source
# --------------------------------------------------------------
git clone https://github.com/pytorch/fairseq.git
cd fairseq
python3.10 -m pip install ./
cd $root_dir

echo "Setup completed!"
