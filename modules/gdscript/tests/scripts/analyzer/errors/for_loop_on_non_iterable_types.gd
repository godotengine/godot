class Iterator:
	func do() -> void:
		pass

func test():
	for x in Object.new():
		pass
	for x in Iterator:
		pass
	for x in Iterator.new():
		pass
