import os, sys, random, json
from ctl import Ctl
#with open("config.json", "r") as f: cfg = json.load(f)



class TwitchBotController():
	def __init__(self):
		self.ctl = Ctl(self.update, self.stopped, self.title)
		self.botData = {
			"prompt":"",
			"data": "",
			"scores": {}
		}

	def update(self, txt):
		#self.botData["data"] = txt
		self.setv("data",txt)

	def title(self, txt):
		self.setv("prompt",txt)
		#self.botData["prompt"] = txt

	def stopped(self):
		self.botData["data"] = ""
		self.botData["prompt"] = "Waiting for new prompt..."

	def prompt(self,txt):
		print(txt)
		#self.setv("prompt",txt)
		self.ctl.start(txt)
	
	def newprompt(self):
		self.ctl.stop()
	
	def setv(self,k,v):
		self.botData[k] = v
	
	def get(self, k):
		return self.botData[k]



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


"""

class TwitchBot(): #irc.bot.SingleServerIRCBot
	def __init__(self):
		#self.bot = bot
		#controller = self
		self.controller = controller
	
	def start(self):
		bot.start()
		#return ()


"""