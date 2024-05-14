#!/bin/bash

# This script finetunes the pretrained translation model on the binarized data using fairseq.


echo `date`
exp_dir=indic-en-exp                             # path of the experiment directory
# model_arch=${2:-"transformer_base18L"}
model_arch=${2:-"transformer_18_18"}    # model architecture (defaults to `transformer_18_18`)
# pretrained_ckpt=./indic-en-exp/checkpoint_best.pt                      # path to the pretrained checkpoint `.pt` file
pretrained_ckpt=./data/jaygala/it2_ckpts/base_models/indic-indic/fairseq_model/model/checkpoint_best.pt
# pretrained_ckpt=./ckpt/indic-en-preprint/fairseq_model/model/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=1 fairseq-train $exp_dir/final_bin \
    --max-source-positions=256 \  # Maximum length of source sequences
    --max-target-positions=256 \  # Maximum length of target sequences
    --source-lang=SRC \           # Source language identifier
    --target-lang=TGT \           # Target language identifier
    --max-update=10 \             # Maximum number of updates (batches) during training
    --save-interval-updates=1000 \ # Frequency of saving model checkpoints during training
    --arch=$model_arch \          # Model architecture specification
    --activation-fn gelu \        # Activation function used in the model
    --criterion=label_smoothed_cross_entropy \  # Loss function used during training
    --label-smoothing=0.1 \       # Label smoothing parameter
    --optimizer adam \            # Optimizer used for training
    --adam-betas "(0.9, 0.98)" \  # Beta coefficients for the Adam optimizer
    --lr-scheduler=inverse_sqrt \ # Learning rate scheduler
    --clip-norm 1.0 \             # Clip gradients to a maximum norm
    --warmup-init-lr 1e-07 \      # Initial learning rate during warmup
    --lr 3e-5 \                   # Learning rate
    --warmup-updates 2000 \       # Number of warmup updates
    --dropout 0.2 \               # Dropout probability
    --save-dir $exp_dir/model_pt \ # Directory to save model checkpoints
    --keep-last-epochs 5 \        # Number of previous epochs to keep
    --keep-interval-updates 3 \   # Interval to keep model checkpoints
    --patience 1 \                # Patience for early stopping
    --skip-invalid-size-inputs-valid-test \  # Skip invalid size inputs during validation and testing
    --fp16 \                      # Enable mixed-precision training
    --user-dir model_configs \    # Directory containing additional model configurations
    --update-freq=4 \             # Update frequency
    --distributed-world-size 1 \  # Distributed training world size
    --num-workers 24 \            # Number of workers for data loading
    --max-tokens 1024 \           # Maximum number of tokens per batch
    --eval-bleu \                 # Evaluate BLEU score during validation
    --eval-bleu-args "{\"beam\": 1, \"lenpen\": 1.0, \"max_len_a\": 1.2, \"max_len_b\": 10}" \  # Arguments for BLEU evaluation
    --eval-bleu-detok moses \     # Detokenizer for BLEU evaluation
    --eval-bleu-remove-bpe sentencepiece \  # Remove BPE tokens for BLEU evaluation
    --eval-bleu-print-samples \   # Print sample translations during BLEU evaluation
    --best-checkpoint-metric bleu \  # Metric for selecting the best checkpoint
    --maximize-best-checkpoint-metric \  # Maximize the best checkpoint metric
    --restore-file $pretrained_ckpt \   # Pretrained checkpoint to restore
    --reset-lr-scheduler \        # Reset learning rate scheduler
    --reset-meters \              # Reset meters
    --reset-dataloader \          # Reset data loader
    --reset-optimizer \           # Reset optimizer
    --task translation            # Task for the model
