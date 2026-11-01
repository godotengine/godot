class AnonRef:
	extends RefCounted
	func fun() -> void:
		pass
	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("    AnonRef released")

static func _receiver(fun1: Callable, fun2: Callable, fun3: Callable) -> void:
	print("    ", fun1.get_object() != null)
	print("    ", fun2.get_object() != null)
	print("    ", fun3.get_object() != null)

func test() -> void:
	_receiver(
		AnonRef.new().fun,
		AnonRef.new().fun,
		AnonRef.new().fun)
	print("    end of statement")
