# https://github.com/godotengine/godot/issues/54589
# https://github.com/godotengine/godot/issues/56265

extends Resource

func test():
	print("okay")
	await self.changed
	await unknown(self)

func unknown(arg):
	await arg.changed
