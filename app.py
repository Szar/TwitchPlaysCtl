import os, sys, irc.bot, random, time, json, requests, threading, re
from ctl import Ctl

with open("config.json", "r") as f: cfg = json.load(f)

#print("Token Auth URL: ")
#print("https://id.twitch.tv/oauth2/authorize?client_id="+cfg["twitch"]["client_id"]+"&redirect_uri="+cfg["twitch"]["redirect_uri"]+"&response_type=token&scope=chat:edit%20chat:read%20user:read:email%20user:read:broadcast%20channel:read:subscriptions%20bits:read%20analytics:read:games")

class TwitchController():
	def __init__(self, say):
		self.ctl = Ctl(self.update, self.stopped, self.title)
		self.prompt = {}
		self.running = False
		self.guess = {}
		self.say = say

	def addGuess(self,message):
		guess = re.sub(r'\W+', '', message["command_text"].lower().strip()).strip()
		if guess!="" and guess!=" ":
			self.guess[message["username"]] = guess
			#print(self.guess)
			if "id" in self.prompt:
				r = requests.post(cfg["defaults"]["api_url"]+"?action=update_guess", data={"guess":json.dumps(self.guess), "id":self.prompt["id"]})
				print("=== guess ===")
				print(r.text)
			#.json()

	def addPrompt(self,message):
		prompt = message["command_text"]
		r = requests.post(cfg["defaults"]["api_url"]+"?action=add_prompt", data=message).json()
		print("=== addPrompt ===")
		print(json.dumps(r))
		return r[0]
	
	def getUser(self,message):
		try:
			r = requests.post(cfg["defaults"]["api_url"]+"?action=user", data=message).json()
		except:
			r = {}
		return r

	def getCurrentPrompt(self):
		r = requests.get(cfg["defaults"]["api_url"]+"?action=get_prompt").json()
		print("=== getCurrentPrompt ===")
		print(json.dumps(r))

		return r

	def startPrompt(self, message):
		if self.running is False:
			self.running = True
			self.prompt = self.addPrompt(message)
			if __name__ == "__main__":
				self.ctl.setPrompt(self.prompt["prompt"])
				self.thr = threading.Thread(target=self.ctl.start)
				self.thr.start()


	def stopPrompt(self):
		self.ctl.running = False
		self.ctl.stop()
		self.thr.join()
	
	def score(self, txt, word):
		
		winners = []
		try:
			if len(self.guess)>0:
				for u in self.guess:
					if word==self.guess[u]:
						winners.append(u)
						r = requests.get(cfg["defaults"]["api_url"]+"?action=add_score&username="+u+"&score=1&channel=unknown").json()

				if len(winners)>0:
					self.say(", ".join(winners)+" guessed the word \""+word+"\" correctly!")
				else:
					self.say("The correct word was \""+word+"\"")
		except Exception as e:
			print(e)

		self.guess = {}

		
	def update(self, txt):
		self.prompt["text"] = txt
		word = re.sub(r'\W+', '', txt.strip().split(" ")[-1].lower().strip()).strip()
		r = requests.post(cfg["defaults"]["api_url"]+"?action=update_prompt", data={"text":txt, "id":self.prompt["id"], "word": word})
		print("=== update ===")
		print(txt)
		self.score(txt, word)

	def stopped(self):
		self.running = False
		r = requests.get(cfg["defaults"]["api_url"]+"?action=deactivate").json()
		print("=== stopPrompt ===")
		print(json.dumps(r))

	def title(self, txt):
		pass

class TwitchBot(irc.bot.SingleServerIRCBot):
	def __init__(self, username, client_id, token, channel):
		self.client_id = client_id
		self.token = token
		self.channel = '#' + channel
		self.controller = TwitchController(self.say)
		
		r = requests.get('https://api.twitch.tv/kraken/users?login=' + channel, headers={'Client-ID': client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}).json()
		self.channel_id = r['users'][0]['_id']
		
		irc.bot.SingleServerIRCBot.__init__(self, [('irc.chat.twitch.tv', 6667, 'oauth:'+token)], username, username)
		print("[ Connected to Twitch ]") 
		


	def on_welcome(self, c, e):
		c.cap('REQ', ':twitch.tv/membership')
		c.cap('REQ', ':twitch.tv/tags')
		c.cap('REQ', ':twitch.tv/commands')
		c.join(self.channel)
		print("[ Bot Running ]") 
		self.controller.stopped()
		
	def on_pubmsg(self, c, e):
		if e.arguments[0][:1] == '!':
			cmd = e.arguments[0].split(' ')[0][1:]
			print('[ Received command: ' + cmd+ ' ]')
			self.do_command(e, cmd)
		return

	def set_message(self,e,cmd):
		tag_keys = ["display-name","id","mod","subscriber","turbo","user-id","user-type"] #badges, flags
		message = { "channel": e.target, "text": e.arguments[0].strip(), "command": cmd, "command_text": e.arguments[0].strip().replace("!"+cmd+" ",""), "username": e.source.split("!")[0] }
		for t in e.tags:
			if t["key"] in tag_keys:
				message[t["key"]] = t["value"]
		return message

	def say(self,m):
		self.connection.privmsg(self.channel, "[Ctl] "+m)

	def do_command(self, e, cmd):
		c = self.connection
		message = self.set_message(e, cmd)
		user = self.controller.getUser(message)
		
		if cmd == "prompt":
			self.controller.startPrompt(message)
			c.privmsg(self.channel, "[Ctl] Using prompt \""+message["command_text"]+"\"")
		
		elif cmd == "new-prompt":
			self.controller.stopPrompt()
			c.privmsg(self.channel, "[Ctl] Ready for new prompt.")

		elif cmd == "guess":
			self.controller.addGuess(message)
			#c.privmsg(self.channel, "[Ctl] "+message["username"]+" thinks the next word will be \""+message["command_text"]+"\"")


if __name__ == "__main__":
	bot = TwitchBot(cfg["twitch"]["bot_username"], cfg["twitch"]["client_id"], cfg["twitch"]["bot_token"], cfg["twitch"]["channel"])
	bot.start()