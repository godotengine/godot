func test() -> void:
	var condition: bool = true

	var data: Dictionary[String, int] = {
		old = 0,
		new = 1,
	} if condition else {
		old = 8,
		new = 9,
	}

	Utils.check(data.get_typed_key_builtin() == TYPE_STRING)
	Utils.check(data.get_typed_value_builtin() == TYPE_INT)
	Utils.check(data["old"] == 0)
	Utils.check(data["new"] == 1)
	print("ok")
