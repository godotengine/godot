func untyped_func(_x: Variant) -> void:
	pass

func array_func(_x: Array) -> void:
	pass

func typed_array_func(_x: Array[int]) -> void:
	pass

func return_array(_pba: PackedByteArray) -> Array:
	return _pba

func declare_array(_pba: PackedByteArray) -> void:
	var _array: Array = _pba

func assign_array(_pba: PackedByteArray) -> void:
	var _array: Array = []
	_array = _pba

func constant_array() -> void:
	const _pba: PackedByteArray = [1]
	const _array: Array = _pba

func test():
	var pba := PackedByteArray()
	var psa := PackedStringArray()
	untyped_func(pba)
	array_func(pba)
	typed_array_func(pba)
	array_func(psa)
	var _returned := return_array(pba)
	declare_array(pba)
	assign_array(pba)
	constant_array()
