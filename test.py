from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_eager_execution()
import transformer, argparse, pdb, sys, re, fastBPE, platform, random, time, json
from collections import Counter
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from control_codes import CONTROL_CODES
from multiprocessing import Pool
with open("config.json", "r") as f: cfg = json.load(f)

generate_num = cfg["defaults"]["generate_num"]
temperature = cfg["defaults"]["temperature"]
split_prompt = ""
text = ""
tf.compat.v1.random.set_random_seed(cfg["defaults"]["seed"])
os.environ['PYTHONHASHSEED'] = str(cfg["defaults"]["seed"])
np.random.seed(cfg["defaults"]["seed"])
vocab = open('vocab', encoding='utf-8').read().split('\n')
vocab = list(map(lambda x: x.split(' ')[0], vocab)) + ['<unk>'] + ['\n']
vocab_size = len(vocab)
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)
seq_length = min(generate_num, 256)
embedding_dim = 1280

bpe = fastBPE.fastBPE('codes', 'vocab')

while True:
	i = input("Prompt: ")
	p = "Movies "+i
	prompt = p.replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ").strip().split('\\n')
	split_prompt = ' \n '.join(bpe.apply(prompt)).replace("","")
	split_prompt = split_prompt.split(' ')
	print(split_prompt)
	if not any(split_prompt[0] == x for x in CONTROL_CODES.keys()):
		print("WARNING! You are not starting your generation from a control code so you won't get good results")
	