class A:
	func return_variant() -> Variant: return null
	func return_void() -> void: pass
	func return_int() -> int: return 123
	func return_int_array(_string_array: Array[String]) -> Array[int]: return []
	func return_int_dict(_string_dict: Dictionary[String, String]) -> Dictionary[int, int]: return {}
	func return_node(_resource: Resource) -> Node: return null

class B extends A:
	func return_variant(): pass
	func return_void(): return 1
	func return_int(): return "abc"
	func return_int_array(string_array: Array[String]): return string_array
	func return_int_dict(string_dict: Dictionary[String, String]): return string_dict
	func return_node(resource: Resource): return resource

func test():
	pass
