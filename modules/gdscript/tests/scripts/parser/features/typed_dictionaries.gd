func test():
	var my_dictionary: Dictionary[int, String] = { 1: "one", 2: "two", 3: "three" }
	var inferred_dictionary := { 1: "one", 2: "two", 3: "three" } # This is Dictionary[int, String].
	print(my_dictionary)
	print(inferred_dictionary)
