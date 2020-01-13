from flask_api import FlaskAPI, status, exceptions
from flask_api.decorators import set_renderers
from multiprocessing import Pool
import os, sys, threading, irc.bot, random, time, json
from twitchplaysbot import TwitchBot

with open("config.json", "r") as f: cfg = json.load(f)

print("Token Auth URL: ")
print("https://id.twitch.tv/oauth2/authorize?client_id="+cfg["twitch"]["client_id"]+"&redirect_uri="+cfg["twitch"]["redirect_uri"]+"&response_type=token&scope=chat:edit%20chat:read%20user:read:email%20user:read:broadcast%20channel:read:subscriptions%20bits:read%20analytics:read:games")


app = FlaskAPI(__name__)
bot = TwitchBot()
thr = threading.Thread(target=bot.start, args=(), kwargs={})
thr.start()
#thread = threading.Thread(target=bot.start)
#thread.start()
#bot.start()
#bot = TwitchPlaysBot(cfg["twitch"]["bot_username"], cfg["twitch"]["client_id"], cfg["twitch"]["bot_token"], cfg["twitch"]["channel"])
#twitchBot = TwitchPlaysBot()

if (sys.version_info > (3, 0)): app.config['DEFAULT_RENDERERS'] = ['flask_api.renderers.JSONRenderer']
else: app.config['DEFAULT_RENDERERS'] = [ 'flask.ext.api.renderers.JSONRenderer', ]

@app.route("/", methods=['GET', 'POST'])
def main():
	return {
			"data":bot.controller.get("data"),
			"scores":bot.controller.get("scores"),
			"prompt":bot.controller.get("prompt")
		}

@app.route("/start", methods=['GET', 'POST'])
def start():
	
	return {"message":"started"}

@app.route("/prompt", methods=['GET', 'POST'])
def prompt():
	return {"data":bot.controller.get("prompt")}

@app.route("/scores", methods=['GET', 'POST'])
def scores():
	return {"data":bot.controller.get("scores")}

@app.route("/data", methods=['GET', 'POST'])
def data():
	return {"data":bot.controller.get("data")}

@app.route("/url_3/<int:pid>/", methods=['GET', 'POST'])
#@set_renderers([renderers.JSONRenderer()])

def url_3(pid):
	return ["url_3 "+str(pid)]

"""
def createBot():
	global bot
	bot = TwitchPlaysBot(cfg["twitch"]["bot_username"], cfg["twitch"]["client_id"], cfg["twitch"]["bot_token"], cfg["twitch"]["channel"])
	thr = threading.Thread(target=bot.start, args=(), kwargs={})
	thr.start()
"""

if __name__ == "__main__":
	app.run(host= '0.0.0.0', port=cfg["defaults"]["port"], debug=True)
	
	#thr = threading.Thread(target=bot.start, args=(), kwargs={})
	