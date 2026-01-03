func test():
	var array: Array = [1, 2, 3]
	print(array)
	var array_clear: Callable = array.clear
	array_clear.call()
	print(array)

	var dictionary: Dictionary = {1: 2, 3: 4}
	print(dictionary)
	# `dictionary.clear` is treated as a key.
	var dictionary_clear := Callable.create(dictionary, &"clear")
	dictionary_clear.call()
	print(dictionary)
