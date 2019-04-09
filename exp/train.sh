#!/bin/bash

nohup python3 train_bilingual_batch.py --config_file ./../conf/bwe2.conf 1>tune2.out 2>err2.out &

nohup python3 train_bilingual_batch.py --config_file ./../conf/bwe3.conf 1>tune3.out 2>err3.out &