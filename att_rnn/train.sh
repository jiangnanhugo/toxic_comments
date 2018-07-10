#!/usr/bin/env bash
python3 main.py --is_train True \
             --batch_size 30 \
             --embd_file ../data/glove.840B.300d.txt.crp \
             --embedding_dim 300 \
             --sequence_length 150 \
             --num_epochs 10
