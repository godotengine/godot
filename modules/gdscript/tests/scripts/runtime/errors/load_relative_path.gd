func test():
	# Relative paths are not supported for function as `Callable`.
	print(load.call("load_relative_path.gd"))
