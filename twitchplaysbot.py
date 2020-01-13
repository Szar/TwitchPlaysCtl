import os, sys, threading, random, time, json, requests
import asynctwitch
#irc.bot
#from pyaib.ircbot import IrcBot
from multiprocessing import Pool
from ctl import Ctl
with open("config.json", "r") as f: cfg = json.load(f)

bot = asynctwitch.CommandBot(
			user = cfg["twitch"]["bot_username"],
			oauth = 'oauth:'+cfg["twitch"]["bot_token"],  # oauth:1234567890abcdefghijklmnopqrst
			channel = cfg["twitch"]["channel"],     
			prefix = "!",  
		)
botData = {
	"prompt":"",
	"data": "",
	"scores": {}
}

class Controller():
	def __init__(self):
		self.client_id =  cfg["twitch"]["client_id"]
		self.token = cfg["twitch"]["bot_token"]
		self.channel = '#' + cfg["twitch"]["channel"]
		self.ctl = Ctl(self.update)

	def update(self, txt):
		botData["data"] = txt

	def prompt(self,txt):
		self.setv("prompt",txt)
		self.ctl.prompt(txt)
	
	def newprompt(self):
		print("restarting...")
	
	def setv(self,k,v):
		global botData
		botData[k] = v
	
	def get(self, k):
		global botData
		return botData[k]



	"""

	def do_command(self, e, cmd):
		c = self.connection
		print(cmd)
		if cmd == "new-prompt":
			c.privmsg(self.channel, str(is_running))
			#if is_running:
			#	c.privmsg(self.channel, "[IncelBot] Restarting... please wait.")
			#	stopgen()
			#	c.privmsg(self.channel, "[IncelBot] Waiting for new prompt.")
		elif cmd == "prompt":
			print("yes!")
			c.privmsg(self.channel, "yes!")
			#if is_running is False:
			#	txt = e.arguments[0].replace("!prompt ","").strip()
			#	if txt!="":
			#		c.privmsg(self.channel, "[IncelBot] Using prompt \""+txt+"\"")
			#		rungen(txt)
		#elif cmd == "seed":
			#setSeed(int(e.arguments[0].replace("!seed ","").strip()))"""


controller = Controller()

class TwitchBot(): #irc.bot.SingleServerIRCBot
	def __init__(self):
		#self.bot = bot
		#controller = self
		self.controller = controller
	
	def start(self):
		bot.start()
		#return ()


@bot.command('example', alias=['moreexample','anothaone'], desc='example command')
async def example(message, word1:str, number1:int, rest:str):
	bot.say(message.channel.name, 'wow')


@bot.command('new-prompt', alias=[], desc='Restart prompt')
async def say(message, subcommand:str):
	controller.newprompt()
	#print(botData)
	#bot.say('yes!')

@bot.command('prompt', alias=[], desc='Set new prompt')
async def say(message, subcommand:str):
	controller.prompt(subcommand)
	#print(botData)
