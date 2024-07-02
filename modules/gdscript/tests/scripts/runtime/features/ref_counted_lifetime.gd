# https://github.com/godotengine/godot/issues/86788

extends Node

class MyClassTyped:
	func enter(outer):
		outer.my_class_typed = MyClassTyped.new()
		exit()

	func exit():
		pass

class MyClassUnTyped:
	func enter(outer):
		outer.my_class_untyped = MyClassUnTyped.new()
		exit()

	func exit():
		pass

var my_class_typed := MyClassTyped.new()
var my_class_untyped = MyClassUnTyped.new()

func test():
	my_class_typed.enter(self)
	my_class_untyped.enter(self)
