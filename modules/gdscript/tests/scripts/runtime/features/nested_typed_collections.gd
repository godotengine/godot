func test():
	var arr: Array[Array[int]] = [[1, 2], [3, 4]]
	arr[0].append(99)
	print(arr)

	var dict: Dictionary[String, Array[int]] = {"a": [1, 2]}
	dict["b"] = [3, 4]
	print(dict)

	var deep: Dictionary[String, Dictionary[String, int]] = {"root": {"x": 1}}
	deep["root"]["y"] = 2
	print(deep)

	print("ok")
