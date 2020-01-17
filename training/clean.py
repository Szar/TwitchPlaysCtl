
#!/usr/bin/python
# -*- coding: utf-8 -*-

filename = input("File: ")

def openFile(fn):
	f = open(fn,'r',encoding="utf8")
	d = f.read()
	f.close()
	return d

def saveFile(fn,d):
	f = open(fn,'w',encoding="utf8")
	f.write(d)
	f.close()

content = openFile(filename)
content = content.replace("  ", " ").replace("\r ", "\r").replace(" \r", "\r").replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'").strip()

saveFile(filename, content)