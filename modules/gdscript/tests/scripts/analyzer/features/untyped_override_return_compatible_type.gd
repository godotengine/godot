class A:
	func return_float() -> float: return 1.0
	func return_int_as_float(_x: int) -> float: return 1.0
	func return_variant_as_float(_x: Variant) -> float: return 1.0
	func return_float_array() -> Array[float]: return [1.0]
	func return_float_dict() -> Dictionary[float, float]: return {1.0: 1.0}

class B extends A:
	func return_float(): return 2
	func return_int_as_float(x: int): return x
	func return_variant_as_float(x: Variant): return x
	func return_float_array(): return [2]
	func return_float_dict(): return {2: 2}

func output(value: Variant) -> void:
	print(var_to_str(value).replace("\n", ""))

func test():
	var b := B.new()

	output(b.return_float())
	output(b.return_int_as_float(2))
	output(b.return_variant_as_float(2))
	output(b.return_float_array())
	output(b.return_float_dict())
