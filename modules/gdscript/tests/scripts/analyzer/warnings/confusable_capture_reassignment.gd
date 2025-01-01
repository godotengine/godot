var member := 1

func test():
	var number := 1
	var string := "1"
	var vector := Vector2i(1, 0)
	var array_assign := [1]
	var array_index := [1]
	var dictionary := { x = 0 }

	var lambda := func ():
		member = 2 # Member variable, not captured.
		number = 2 # Local variable, captured.
		string += "2" # Test compound assignment operator.
		vector.x = 2 # Test subscript assignment.
		array_assign = [2] # Pass-by-reference type, reassignment.
		array_index[0] = 2 # Pass-by-reference type, index access.
		dictionary.x = 2 # Pass-by-reference type, attribute access.

		prints("lambda", member, number, string, vector, array_assign, array_index, dictionary)

	lambda.call()
	prints("outer", member, number, string, vector, array_assign, array_index, dictionary)
