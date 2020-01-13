from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_eager_execution()
import transformer, argparse, pdb, sys, re, threading, fastBPE, platform, irc.bot, requests, random, time, json
from collections import Counter
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from control_codes import CONTROL_CODES
from multiprocessing import Pool
import config as cfg
with open("config.json", "r") as f: cfg = json.load(f)

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type=str, required=True, help='location of model checkpoint. E.g. "YOUR_CKPT_DIR.ckpt"')

print("Token Auth URL: ")
print("https://id.twitch.tv/oauth2/authorize?client_id="+cfg.twitch["client_id"]+"&redirect_uri="+cfg.twitch["redirect_uri"]+"&response_type=token&scope=chat:edit%20chat:read%20user:read:email%20user:read:broadcast%20channel:read:subscriptions%20bits:read%20analytics:read:games")

args = parser.parse_args()
model_dir = args.model_dir
aseed = cfg["defaults"]["seed"]
generate_num = cfg["defaults"]["generate_num"]
temperature = cfg["defaults"]["temperature"]
nucleusprob = cfg["defaults"]["nucleusprob"]
topk = cfg["defaults"]["topk"]
penalty = cfg["defaults"]["penalty"]
topn = cfg["defaults"]["topn"]
prompt = ""
split_prompt = ""
text = ""
is_running = False


tf.compat.v1.random.set_random_seed(aseed)
os.environ['PYTHONHASHSEED'] = str(aseed)
np.random.seed(aseed)

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
		model_dir=model_dir)

estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)

def serving_input_fn():
	inputs = {'input_1': tf.compat.v1.placeholder(tf.int32, [1,seq_length])}
	return tf.estimator.export.ServingInputReceiver(inputs, inputs)
predict_fn = tf.contrib.predictor.from_estimator(estimator_model, serving_input_fn)
bpe = fastBPE.fastBPE('codes', 'vocab')


def setPrompt(txt):
	global prompt
	global split_prompt
	prompt = "Pill "+txt
	prompt = prompt.split('\\n')
	split_prompt = ' \n '.join(bpe.apply(prompt))
	split_prompt = split_prompt.split(' ')
	if not any(split_prompt[0] == x for x in CONTROL_CODES.keys()):
		print("WARNING! You are not starting your generation from a control code so you won't get good results")
	print()
	print()
	print("----- "+"["+split_prompt[0]+"] \""+txt+"\" -----")
	return prompt

thr = threading.Thread(target=setPrompt, args=(), kwargs={})

def gentext():
	global text
	global is_running
	global thr
	text = [word2idx[i] for i in split_prompt]
	padded_text = text + [0] * (generate_num - len(text))
	tokens_generated = np.tile(padded_text, (1,1))
	
	brstring = "                               "
	try:
		for token in range(len(text)-1, generate_num-1):
			if is_running is False:
				token = generate_num
				break
			else:
				if is_running:
					if token <= seq_length:
						prompt_logits = predict_fn({'input_1':tokens_generated[:, :seq_length]})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)
						_token = token if token < seq_length else -1
					else:
						_token = -1
						end = token + 1
						start = token - seq_length + 2
						prompt_logits = predict_fn({'input_1':np.hstack((tokens_generated[:,0:1], tokens_generated[:,start:end]))})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)

					if penalty>0:
						penalized_so_far = set()
						for _ in range(token+1):
							generated_token = tokens_generated[0][_]
							if idx2word[generated_token] == '\n':
								continue
							if generated_token in penalized_so_far:
								continue
							penalized_so_far.add(generated_token)
							prompt_logits[_token][generated_token] /= penalty

					# disallow some tokens
					prompt_logits[_token][word2idx['<unk>']] = -1e8
					prompt_logits[_token][word2idx['Sco@@']] = -1e8

					prompt_probs = np.exp(prompt_logits[_token])
					prompt_probs = prompt_probs / sum(prompt_probs)
					pruned_list = np.argsort(prompt_probs)[::-1]
					if nucleusprob > 0.:
						minimum_topk = 1
						nucleus = max(np.where(np.cumsum(np.sort(prompt_probs)[::-1])>nucleusprob)[0][0], minimum_topk)
					elif topk > 0:
						nucleus = topk
					else:
						nucleus = len(pruned_list)
						
					pruned_list = pruned_list[:nucleus]  

					tokens_to_disallow = []
					for _ in range(len(pruned_list)):
						if 'http' in idx2word[pruned_list[_]]:
							tokens_to_disallow.append(_)
					pruned_list = np.delete(pruned_list, tokens_to_disallow)

					if topn > 0 :
						print('TOPN :: top-n alternatives:', [idx2word[_] for _ in pruned_list[:topn]])

					if temperature==0:
						idx = pruned_list[0]
					else:
						chosen_idx = int(tf.random.categorical(np.expand_dims(prompt_logits[_token][pruned_list],0), num_samples=1).numpy())
						idx = pruned_list[chosen_idx]

					if topn > 0 :
						print('TOPN :: chosen word:', idx2word[idx])

					tokens_generated[0][token+1] = idx

					tokens_generated_so_far = ' '.join([idx2word[c] for c in tokens_generated[0].squeeze()[:token+2]])
					tokens_generated_so_far = re.sub('(@@ )', '', string=tokens_generated_so_far)
					tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)              
  
				brstring = " "
				for c in tokens_generated_so_far:
					brstring+=" "

				print()
				print()
				
				print(tokens_generated_so_far[5:]) 
				print()
				print()
				print("-")

	except KeyboardInterrupt:
		return
	print()
	print()
	print("Awaiting new prompt...") 
	return


def rungen(txt):
	global is_running
	global thr
	is_running = True
	
	
	setPrompt(txt)
	thr = threading.Thread(target=gentext, args=(), kwargs={})
	thr.start()
	

def stopgen():
	global is_running
	global thr
	is_running = False
	thr.join()
	return
	
def setSeed(s):
	global seed
	seed = s

class TwitchBot(irc.bot.SingleServerIRCBot):
	def __init__(self, username, client_id, token, channel):
		self.client_id = client_id
		self.token = token
		self.channel = '#' + channel
		url = 'https://api.twitch.tv/kraken/users?login=' + channel
		headers = {'Client-ID': client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}
		r = requests.get(url, headers=headers).json()
		self.channel_id = r['users'][0]['_id']
		irc.bot.SingleServerIRCBot.__init__(self, [('irc.chat.twitch.tv', 6667, 'oauth:'+token)], username, username)
		print('Connected to twitch.')


	def on_welcome(self, c, e):
		c.cap('REQ', ':twitch.tv/membership')
		c.cap('REQ', ':twitch.tv/tags')
		c.cap('REQ', ':twitch.tv/commands')
		c.join(self.channel)
		print()
		print()
		print("Awaiting new prompt...") 

	def on_pubmsg(self, c, e):
		if e.arguments[0][:1] == '!':
			cmd = e.arguments[0].split(' ')[0][1:]
			self.do_command(e, cmd)
		return
	
	def get_user(self, c):
		url = 'https://api.twitch.tv/kraken/channels/' + c
		headers = {'Client-ID': self.client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}
		return requests.get(url, headers=headers).json()


	def do_command(self, e, cmd):
		c = self.connection
		if cmd == "new-prompt":
			c.privmsg(self.channel, str(is_running))
			if is_running:
				c.privmsg(self.channel, "[IncelBot] Restarting... please wait.")
				stopgen()
				c.privmsg(self.channel, "[IncelBot] Waiting for new prompt.")
		elif cmd == "prompt":
			if is_running is False:
				txt = e.arguments[0].replace("!prompt ","").strip()
				if txt!="":
					c.privmsg(self.channel, "[IncelBot] Using prompt \""+txt+"\"")
					rungen(txt)
		elif cmd == "seed":
			setSeed(int(e.arguments[0].replace("!seed ","").strip()))

def main():
	bot = TwitchBot(cfg["twitch"]["bot_username"], cfg["twitch"]["client_id"], cfg["twitch"]["bot_token"], cfg["twitch"]["channel"])
	bot.start()

if __name__ == "__main__":
	main()

