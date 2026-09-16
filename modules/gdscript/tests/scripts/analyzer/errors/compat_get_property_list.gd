# GH-118877

class Test1:
	func _get_property_list() -> Array[int]:
		return []

class Test2:
	func _get_property_list() -> int:
		return 0

class Test3:
	func _get_property_list() -> void:
		pass

class Test4:
	func _get_property_list():
		return 0

func test():
	pass
