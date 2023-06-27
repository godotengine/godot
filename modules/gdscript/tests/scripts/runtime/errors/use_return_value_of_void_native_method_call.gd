func test():
	var obj
	obj = RefCounted.new()
	print(obj.notify_property_list_changed())
