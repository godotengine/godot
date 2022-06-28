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
	var transform = Transform3D()
	transform.basis.x = Vector3(8.0, 9.0, 7.0)
	print(dictionary1)
	print(dictionary2)
	print(array1)
	print(array2)
	print(array3)
	print(transform)
