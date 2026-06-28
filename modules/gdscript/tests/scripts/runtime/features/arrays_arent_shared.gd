# https://github.com/godotengine/godot/issues/48121

func test():
	var x := []
	var y := []
	x.push_back(y)
	print("TEST ARRAY ADD TO SELF: " + str(len(y)))
	x.clear()

	x = Array()
	y = Array()
	x.push_back(y)
	print("TEST ARRAY ADD TO SELF: " + str(len(y)))
	x.clear()

	x = Array().duplicate()
	y = Array().duplicate()
	x.push_back(y)
	print("TEST ARRAY ADD TO SELF: " + str(len(y)))
	x.clear()

	x = [].duplicate()
	y = [].duplicate()
	x.push_back(y)
	print("TEST ARRAY ADD TO SELF: " + str(len(y)))
	x.clear()

	x = Array()
	y = Array()
	x.push_back(y)
	print("TEST ARRAY ADD TO SELF: " + str(len(y)))
	x.clear()
