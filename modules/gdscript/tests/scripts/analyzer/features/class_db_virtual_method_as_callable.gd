class TestClass:
	pass

func test() -> void:
	var free_callable: Callable = free
	print(free_callable)
	var test_class_new_callable: Callable = TestClass.new
	print(test_class_new_callable)
