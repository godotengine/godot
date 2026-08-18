class Check extends Node:
	@override
	func _set(_property: StringName, _value: Variant) -> bool:
		return true

func test() -> void:
	print('OK')
