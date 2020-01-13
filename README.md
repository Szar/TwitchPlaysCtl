# TwitchPlaysCtl

This bot is based on https://github.com/salesforce/ctrl

### Install Dependencies 

[TensorFlow 1.14](https://www.tensorflow.org/install)
```
pip install tensorflow==1.14
```
[fastBPE](https://github.com/glample/fastBPE)
```
git clone https://github.com/glample/fastBPE.git
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
python setup.py install
```

### Download Model 
[Download the Model](https://console.cloud.google.com/storage/browser/sf-ctrl/?pli=1) OR run:
```
gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt/ .
```

### Creating TFRecords
From the training/ directory, run:

```
python make_tf_records.py --text_file TRAINING_TEXT_FILE.txt --control_code YOUR_CTL_CODE --sequence_len 256
```

### Training
From the training/ directory, run:
```
python training.py --model_dir ../YOUR_CKPT_DIR.ckpt/ --iterations 100
```

### Configure
Create config.json add:
```
{
	"defaults": {
		"port": 3100,
		"seed": 34391,
		"temperature": 0.9,
		"generate_num": 256,
		"penalty": 1.2,
		"nucleusprob":0,
		"topk":0,
		"topn":0
	},
	"twitch":{
		"client_id":"",
		"redirect_uri":"",
		"bot_username":"",
		"bot_token":"",
		"channel": ""
	}
}
```

Fill out the twitch config with your Twitch app and stream details. In order to generate a token, you can use the following URL:
```
https://id.twitch.tv/oauth2/authorize?client_id=CLIENT_ID_HERE&redirect_uri=REDIRECT_URI_HERE&response_type=token&scope=chat:edit%20chat:read%20user:read:email%20user:read:broadcast%20channel:read:subscriptions%20bits:read%20analytics:read:games
```

### Running
```
python twitchplaysctl.py --model_dir YOUR_CKPT_DIR.ckpt
```
