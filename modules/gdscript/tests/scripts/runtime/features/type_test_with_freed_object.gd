func test():
	var obj := Object.new()
	print(obj is Object)
	obj.free()
	print(obj is Object)
