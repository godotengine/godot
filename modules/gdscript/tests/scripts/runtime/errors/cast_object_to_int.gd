func test():
	var object: Variant = RefCounted.new()
	@warning_ignore("unsafe_cast")
	print(object as int)
