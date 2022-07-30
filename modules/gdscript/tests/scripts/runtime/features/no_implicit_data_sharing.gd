func test():
	var dictionary1: Variant = {1:Vector2()}
	dictionary1[1].x = 2

	var dictionary2: Dictionary = {3:Vector2()}
	dictionary2[3].x = 4

	var array1: Variant = [[Vector2()]]
	array1[0][0].x = 5

	var array2: Array = [[Vector2()]]
	array2[0][0].x = 6

	var array3: Array[Array] = [[Vector2()]]
	array3[0][0].x = 7

	var dictionary3: Dictionary = {3:Vector2()}

	var array4 := []
	var array5 := []
	array4.push_back(array5)

	var array6 := [Array()]
	array6[0].push_back([[Vector2()]])

	var array7 := [[Vector3()]]
	array7[0][0].y = 13
	var array8 := [[Vector3()]]

	print(dictionary1)
	print(dictionary2)
	print(array1)
	print(array2)
	print(array3)
	print(dictionary3)
	print(str(len(array4)))
	print(str(len(array5)))
	print(array6)
	print(array7)
	print(array8)

	# TODO: this one fails since constants act like static vars
	# for i in range(4):
	# 	const x := []
	# 	x.append(i)
	# 	print(x)
