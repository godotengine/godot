func test():
	var _sub = SubClass.new("test")

class SuperClass:
	func _init(param):
		prints("SuperClass init", param)

class SubClass extends SuperClass:
	# No _init implementation, will use the superclass one.
	pass
