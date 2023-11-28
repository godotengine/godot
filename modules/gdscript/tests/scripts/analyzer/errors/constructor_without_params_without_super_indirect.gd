# https://github.com/godotengine/godot/issues/40861

func test():
	var _sub = SubSubClass.new()

class SuperClass:
	func _init():
		print("No parameters")

class SubClass extends SuperClass:
	# No custom constructor.
	pass

class SubSubClass extends SubClass:
	func _init():
		print("Missing super()")
