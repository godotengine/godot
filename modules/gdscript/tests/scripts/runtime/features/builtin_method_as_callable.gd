func test():
	var array: Array = [1, 2, 3]
	print(array)
	var callable: Callable = array.clear
	callable.call()
	print(array)
