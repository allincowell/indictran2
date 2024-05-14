#!/bin/bash

# This script finetunes the pretrained translation model on the binarized data using fairseq.


echo `date`
exp_dir=$1                             # path of the experiment directory
# model_arch=${2:-"transformer_base18L"}
model_arch=${2:-"transformer_18_18"}    # model architecture (defaults to `transformer_18_18`)
# pretrained_ckpt=./indic-en-exp/checkpoint_best.pt                      # path to the pretrained checkpoint `.pt` file
pretrained_ckpt=./data/jaygala/it2_ckpts/base_models/indic-indic/fairseq_model/model/checkpoint_best.pt
# pretrained_ckpt=./ckpt/indic-en-preprint/fairseq_model/model/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=1 fairseq-train $exp_dir/final_bin \
--max-source-positions=256 \
--max-target-positions=256 \
--source-lang=SRC \
--target-lang=TGT \
--max-update=2000000 \
--save-interval-updates=1000 \
--arch=$model_arch \
--activation-fn gelu \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--lr-scheduler=inverse_sqrt \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 3e-5 \
--warmup-updates 2000 \
--dropout 0.2 \
--save-dir $exp_dir/model_pt \
--keep-last-epochs 5 \
--keep-interval-updates 3 \
--patience 1 \
--skip-invalid-size-inputs-valid-test \
--fp16 \
--user-dir model_configs \
--update-freq=4 \
--distributed-world-size 1 \
--num-workers 24 \
--max-tokens 1024 \
--eval-bleu \
--eval-bleu-args "{\"beam\": 1, \"lenpen\": 1.0, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe sentencepiece \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--restore-file $pretrained_ckpt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer \
--task translation
