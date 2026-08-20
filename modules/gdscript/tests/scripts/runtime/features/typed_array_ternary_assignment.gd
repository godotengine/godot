func test() -> void:
	var condition: bool = true
	var data: Array[int] = []
	data = [0,1] if condition else [8,9]

	Utils.check(data.get_typed_builtin() == TYPE_INT)
	Utils.check(data[0] == 0)
	Utils.check(data[1] == 1)
	print("ok")
