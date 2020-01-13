from flask_api import FlaskAPI, status, exceptions
from flask_api.decorators import set_renderers
from multiprocessing import Pool
import os, sys, threading, irc.bot, random, time, json
from twitchplaysbot import TwitchBotController
import asynctwitch

with open("config.json", "r") as f: cfg = json.load(f)

print("Token Auth URL: ")
print("https://id.twitch.tv/oauth2/authorize?client_id="+cfg["twitch"]["client_id"]+"&redirect_uri="+cfg["twitch"]["redirect_uri"]+"&response_type=token&scope=chat:edit%20chat:read%20user:read:email%20user:read:broadcast%20channel:read:subscriptions%20bits:read%20analytics:read:games")


app = FlaskAPI(__name__)
twitchplaysbot = TwitchBotController()

bot = asynctwitch.CommandBot(
			user = cfg["twitch"]["bot_username"],
			oauth = 'oauth:'+cfg["twitch"]["bot_token"],  # oauth:1234567890abcdefghijklmnopqrst
			channel = cfg["twitch"]["channel"],     
			prefix = "!",  
		)
@bot.command('example', alias=['moreexample','anothaone'], desc='example command')
async def example(message, word1:str, number1:int, rest:str):
	bot.say(message.channel.name, 'wow')


@bot.command('new-prompt', alias=[], desc='Restart prompt')
async def newprompt(message, subcommand:str):
	twitchplaysbot.newprompt()
	#bot.say('restarting...')

@bot.command('prompt', alias=[], desc='Set new prompt')
async def prompt(message, subcommand:str):
	twitchplaysbot.prompt(subcommand)

thr = threading.Thread(target=bot.start, args=(), kwargs={})
thr.start()

if (sys.version_info > (3, 0)): app.config['DEFAULT_RENDERERS'] = ['flask_api.renderers.JSONRenderer']
else: app.config['DEFAULT_RENDERERS'] = [ 'flask.ext.api.renderers.JSONRenderer', ]

@app.route("/", methods=['GET', 'POST'])
def main():
	return {
			"data":twitchplaysbot.get("data"),
			"scores":twitchplaysbot.get("scores"),
			"prompt":twitchplaysbot.get("prompt")
		}

if __name__ == "__main__":
	app.run(host= '0.0.0.0', port=cfg["defaults"]["port"], debug=False)
	
	#thr = threading.Thread(target=bot.start, args=(), kwargs={})
	