# https://github.com/godotengine/godot/issues/121316

class Tracked extends RefCounted:
	func return_self() -> Tracked:
		return self

	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("released")

func with_untyped_default(_values: Array = []) -> void:
	Tracked.new().return_self()
	print("after untyped")

func with_typed_default(_values: Array[String] = []) -> void:
	Tracked.new().return_self()
	print("after typed")

func test() -> void:
	with_untyped_default()
	with_typed_default()
