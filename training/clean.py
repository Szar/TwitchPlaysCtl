
filename = input("File: ")

def openFile(fn):
	f = open(fn,'r')
	d = f.read()
	f.close()
	return d

def saveFile(fn):
	f = open(fn,'w')
	f.write(content)
	f.close()

content = openFile(filename)
content = content.replace("  ", " ").replace("\r ", "\r").replace(" \r", "\r").replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'").strip()

saveFile(filename, content)