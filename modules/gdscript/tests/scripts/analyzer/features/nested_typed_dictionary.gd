func test():
	var dict: Dictionary[String, Dictionary[String, int]] = {"a": {"x": 1}, "b": {"y": 2}}
	var dict_arr: Dictionary[String, Array[int]] = {"nums": [1, 2, 3], "more": [4, 5]}

	var inner: Dictionary[String, int] = dict["a"]
	Utils.check(inner.get_typed_key_builtin() == TYPE_STRING)
	Utils.check(inner.get_typed_value_builtin() == TYPE_INT)
	Utils.check(inner["x"] == 1)

	var nums: Array[int] = dict_arr["nums"]
	Utils.check(nums.get_typed_builtin() == TYPE_INT)
	Utils.check(str(nums) == "[1, 2, 3]")

	print("ok")
