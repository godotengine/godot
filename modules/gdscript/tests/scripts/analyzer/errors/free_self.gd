extends Node

func test():
	var l = func lambda():
		self.free()

	self.free()
