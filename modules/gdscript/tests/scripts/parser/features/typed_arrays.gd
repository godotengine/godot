func test():
	var my_array: Array[int] = [1, 2, 3]
	var inferred_array := [1, 2, 3]  # This is Array[int].
	print(my_array)
	print(inferred_array)
