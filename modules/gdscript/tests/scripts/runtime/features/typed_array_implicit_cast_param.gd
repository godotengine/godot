# GH-93990

func test_param(array: Array[String]) -> void:
	print(array.get_typed_builtin() == TYPE_STRING)

func test() -> void:
	test_param(PackedStringArray())
