func test_param(dictionary: Dictionary[int, String]) -> void:
	print(dictionary.get_typed_key_builtin() == TYPE_INT)
	print(dictionary.get_typed_value_builtin() == TYPE_STRING)

func test() -> void:
	test_param({ 123: "some_string" })
