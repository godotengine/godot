class A extends Node:
	var l: Node
	func _init(_a:int):
		l = self.duplicate()

class B extends Resource:
	var l: Resource
	func _init(_a:int):
		l = self.duplicate()

func test():
	print("warn")
