class PrivateMembers extends Node:
	signal _private_signal

	enum _PrivateEnum { VALUE }

	const _PRIVATE_CONSTANT = 1

	static var _private_static_variable := 1

	var _private_variable := 1
	var public_variable := 1

	static func _private_static_method() -> void:
		pass

	func _private_method() -> void:
		pass

	func get_private_method() -> Callable:
		return _private_method

	func _ready() -> void:
		pass

	func _to_string() -> String:
		return ""

	class _PrivateInner:
		pass
