class A:
	func return_int(_variant: Variant) -> int: return 123
	func return_int_array(_variant: Variant) -> Array[int]: return [1]
	func return_int_dict(_variant: Variant) -> Dictionary[int, int]: return {1: 1}
	func return_node(_variant: Variant) -> Node: return null

class B extends A:
	func return_int(variant: Variant): return variant
	func return_int_array(variant: Variant): return variant
	func return_int_dict(variant: Variant): return variant
	func return_node(variant: Variant): return variant

func output(value: Variant) -> void:
	print(var_to_str(value).replace("\n", ""))

func test():
	var b := B.new()

	output(b.return_int("abc"))
	output(b.return_int_array("abc"))
	output(b.return_int_dict("abc"))
	output(b.return_node("abc"))
