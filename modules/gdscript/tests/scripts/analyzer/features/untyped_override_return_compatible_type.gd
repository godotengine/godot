class A:
	func return_float() -> float: return 1.0
	func return_int_as_float(_x: int) -> float: return 1.0
	func return_variant_as_float(_x: Variant) -> float: return 1.0
	func return_float_array() -> Array[float]: return [1.0]
	func return_untyped_array_as_float_array(_array: Array) -> Array[float]: return [1.0]
	func return_float_dict() -> Dictionary[float, float]: return {1.0: 1.0}
	func return_untyped_dict_as_float_dict(_dict: Dictionary) -> Dictionary[float, float]: return {1.0: 1.0}
	func return_object_as_node(_object: Object) -> Node: return null

class B extends A:
	func return_float(): return 2
	func return_int_as_float(x: int): return x
	func return_variant_as_float(x: Variant): return x
	func return_float_array(): return [2]
	func return_untyped_array_as_float_array(array: Array): return array
	func return_float_dict(): return {2: 2}
	func return_untyped_dict_as_float_dict(dict: Dictionary): return dict
	func return_object_as_node(object: Object): return object

func output(value: Variant) -> void:
	if value is Object:
		var object: Object = value
		print("<%s>" % object.get_class())
	else:
		print(var_to_str(value).replace("\n", ""))

func test():
	var b := B.new()
	var float_array: Array[float] = [2]
	var float_dict: Dictionary[float, float] = {2: 2}
	var node := Node.new()

	output(b.return_float())
	output(b.return_int_as_float(2))
	output(b.return_variant_as_float(2))
	output(b.return_float_array())
	output(b.return_untyped_array_as_float_array(float_array))
	output(b.return_float_dict())
	output(b.return_untyped_dict_as_float_dict(float_dict))
	output(b.return_object_as_node(node))

	node.free()
