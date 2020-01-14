from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_eager_execution()
import transformer, argparse, pdb, sys, re, threading, fastBPE, platform, random, time, json
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

class TiedEmbeddingSoftmax(tf.keras.layers.Layer):
	def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
		super(TiedEmbeddingSoftmax, self).__init__()
		self.w = self.add_weight(name='w', shape=(vocab_size, embedding_size), initializer='random_normal', trainable=True)
		self.b = self.add_weight(name='b', shape=(vocab_size,), initializer='zeros', trainable=True)

	def call(self, inputs, embed=True):
		if embed:
			dtype = tf.keras.backend.dtype(inputs)
			if dtype != 'int32' and dtype != 'int64':
				inputs = math_ops.cast(inputs, 'int32')
			return embedding_ops.embedding_lookup(self.w, inputs)
		else:
			return tf.tensordot(inputs, tf.transpose(self.w), 1) + self.b


tokens = tf.keras.layers.Input(shape=(seq_length,), dtype='int32')
tied_embedding_softmax = TiedEmbeddingSoftmax()
embedded = tied_embedding_softmax(tokens, embed=True)
transformed = transformer.Encoder()(embedded, training=False)
logits = tied_embedding_softmax(transformed, embed=False)
model = tf.keras.Model(inputs=tokens, outputs=logits)

def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

optimizer = tf.contrib.tpu.CrossShardOptimizer(
	tf.contrib.estimator.clip_gradients_by_norm(
		tf.train.AdagradOptimizer(learning_rate=1e-2), 0.25)
	)        
		
model.compile(optimizer=optimizer, loss=loss)
print(model.summary())

run_config = tf.contrib.tpu.RunConfig(
		model_dir=cfg["defaults"]["model_dir"])

estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)

def serving_input_fn():
	inputs = {'input_1': tf.compat.v1.placeholder(tf.int32, [1,seq_length])}
	return tf.estimator.export.ServingInputReceiver(inputs, inputs)
predict_fn = tf.contrib.predictor.from_estimator(estimator_model, serving_input_fn)

bpe = fastBPE.fastBPE('codes', 'vocab')

class Ctl():
	def __init__(self, cb, stcb, pcb):
		self.prompt =  ""
		#self.thr = threading.Thread(target=self.setPrompt, args=(), kwargs={})
		self.cb = cb
		self.stcb = stcb
		self.pcb = pcb
		self.running = False
		
	def stop(self):
		self.running = False
		return self.thr.join()

	def start(self,txt):
		if self.running:
			print("Already running prompt.")
		else:
			self.raw_prompt = txt
			self.setPrompt()

	def setPrompt(self):
		global split_prompt
		print("------ Setting Prompt -------")
		self.running = True
		txt = self.raw_prompt.replace("@","")
		prompt = cfg["defaults"]["control_code"]+" "+txt
		prompt = prompt.replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ").strip().split('\\n')
		split_prompt = ' \n '.join(bpe.apply(prompt))
		split_prompt = split_prompt.split(' ')
		if not any(split_prompt[0] == x for x in CONTROL_CODES.keys()):
			print("WARNING! You are not starting your generation from a control code so you won't get good results")
		self.prompt = prompt
	
		self.thr = threading.Thread(target=self.gentext, args=(), kwargs={})
		self.thr.start()
		
		self.pcb(" ".join(split_prompt[1:]))
		
		#self.split_prompt = split_prompt
		self.gentext()
		self.running = False
		
	def gentext(self):
		global text
		print("------ Generating Text -------")
		text = [word2idx[i] for i in split_prompt]
		padded_text = text + [0] * (generate_num - len(text))
		tokens_generated = np.tile(padded_text, (1,1))
		
		try:
			for token in range(len(text)-1, generate_num-1):
				if self.running is False:
					token = generate_num
					break
				else:
					if self.running:
						if token <= seq_length:
							prompt_logits = predict_fn({'input_1':tokens_generated[:, :seq_length]})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)
							_token = token if token < seq_length else -1
						else:
							_token = -1
							end = token + 1
							start = token - seq_length + 2
							prompt_logits = predict_fn({'input_1':np.hstack((tokens_generated[:,0:1], tokens_generated[:,start:end]))})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)

						if cfg["defaults"]["penalty"]>0:
							penalized_so_far = set()
							for _ in range(token+1):
								generated_token = tokens_generated[0][_]
								if idx2word[generated_token] == '\n':
									continue
								if generated_token in penalized_so_far:
									continue
								penalized_so_far.add(generated_token)
								prompt_logits[_token][generated_token] /= cfg["defaults"]["penalty"]

						# disallow some tokens
						prompt_logits[_token][word2idx['<unk>']] = -1e8
						prompt_logits[_token][word2idx['Sco@@']] = -1e8

						prompt_probs = np.exp(prompt_logits[_token])
						prompt_probs = prompt_probs / sum(prompt_probs)
						pruned_list = np.argsort(prompt_probs)[::-1]
						if cfg["defaults"]["nucleusprob"] > 0.:
							minimum_topk = 1
							nucleus = max(np.where(np.cumsum(np.sort(prompt_probs)[::-1])>cfg["defaults"]["nucleusprob"])[0][0], minimum_topk)
						elif cfg["defaults"]["topk"] > 0:
							nucleus = cfg["defaults"]["topk"]
						else:
							nucleus = len(pruned_list)
							
						pruned_list = pruned_list[:nucleus]  

						tokens_to_disallow = []
						for _ in range(len(pruned_list)):
							if 'http' in idx2word[pruned_list[_]]:
								tokens_to_disallow.append(_)
						pruned_list = np.delete(pruned_list, tokens_to_disallow)

						if cfg["defaults"]["topn"] > 0 :
							print('TOPN :: top-n alternatives:', [idx2word[_] for _ in pruned_list[:cfg["defaults"]["topn"]]])

						if temperature==0:
							idx = pruned_list[0]
						else:
							chosen_idx = int(tf.random.categorical(np.expand_dims(prompt_logits[_token][pruned_list],0), num_samples=1).numpy())
							idx = pruned_list[chosen_idx]

						if cfg["defaults"]["topn"] > 0 :
							print('TOPN :: chosen word:', idx2word[idx])

						tokens_generated[0][token+1] = idx

						tokens_generated_so_far = ' '.join([idx2word[c] for c in tokens_generated[0].squeeze()[:token+2]])
						tokens_generated_so_far = re.sub('(@@ )', '', string=tokens_generated_so_far)
						tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)              
	
					self.cb(tokens_generated_so_far[5:])


		except KeyboardInterrupt:
			self.running = False
			return
		self.running = False
		return

#ctl = Ctl()