func test():
	var my_dictionary: Dictionary[int, String] = { 1: "one", 2: "two", 3: "three" }
	var inferred_dictionary := { 1: "one", 2: "two", 3: "three" } # This is Dictionary[int, String].
	var keys := my_dictionary.keys()  # Array[Int]
	var values := my_dictionary.values()  # Array[String]
	print(my_dictionary)
	print(inferred_dictionary)
	print(keys[0])
	print(values[2].length())
