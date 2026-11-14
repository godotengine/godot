func main_void_function() -> void:
	return child_void_function()

func child_void_function() -> void:
	print("ok");

func test():
	main_void_function()
