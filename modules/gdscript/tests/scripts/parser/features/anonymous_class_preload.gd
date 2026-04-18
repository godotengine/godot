func test():
	# Anonymous class extending a preloaded script.
	var obj = preload("anonymous_class_preload_base.notest.gd").new():
		func describe():
			return "preload-anon value=" + str(value)
	print(obj.describe())
