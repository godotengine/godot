# https://github.com/godotengine/godot/issues/48121

func test():
	var x := Dictionary()
	var y := Dictionary()
	y[0]=1
	y[1]=1
	y[2]=1
	print("TEST OTHER DICTIONARY: " + str(len(x)))
	x.clear()

	x = Dictionary().duplicate()
	y = Dictionary().duplicate()
	y[0]=1
	y[1]=1
	y[2]=1
	print("TEST OTHER DICTIONARY: " + str(len(x)))
	x.clear()
	return
