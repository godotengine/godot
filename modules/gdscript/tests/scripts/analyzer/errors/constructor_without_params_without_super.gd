# https://github.com/godotengine/godot/issues/40861

func test():
	var _sub = SubClass.new()

class SuperClass:
	func _init():
		print("No parameters")

class SubClass extends SuperClass:
	func _init():
		print("Missing super()")
