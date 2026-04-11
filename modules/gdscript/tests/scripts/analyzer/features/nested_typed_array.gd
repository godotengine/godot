func test():
	var arr: Array[Array[int]] = [[1, 2], [3, 4]]
	var deep: Array[Array[Array[int]]] = [[[1]], [[2, 3]]]

	var element: Array[int] = arr[0]
	Utils.check(element.get_typed_builtin() == TYPE_INT)
	Utils.check(str(element) == "[1, 2]")

	var inner: Array[int] = deep[0][0]
	Utils.check(inner.get_typed_builtin() == TYPE_INT)
	Utils.check(str(inner) == "[1]")

	arr.append([5, 6])
	Utils.check(str(arr) == "[[1, 2], [3, 4], [5, 6]]")

	print("ok")
