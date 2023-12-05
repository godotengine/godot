# GH-73213

func test():
	var object := Object.new() # Not `Object()`.
	print(object.get_class())
	object.free()
