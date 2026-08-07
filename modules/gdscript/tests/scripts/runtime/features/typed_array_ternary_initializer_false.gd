func test() -> void:
	var condition: bool = false
	var data: Array[int] = [0,1] if condition else [8,9]
	Utils.check(data.get_typed_builtin() == TYPE_INT)
	Utils.check(data[0] == 8)
	Utils.check(data[1] == 9)
	print("ok")
