func untyped_func(_x: Variant) -> void:
	pass

func array_func(_x: Array) -> void:
	pass

func typed_array_func(_x: Array[int]) -> void:
	pass

func test():
	var pba := PackedByteArray()
	untyped_func(pba)
	array_func(pba)
	typed_array_func(pba)

