class Something extends RefCounted implements SimSystem:
	func get_value() -> int:
		return 42

class NotASystem extends RefCounted:
	func get_value() -> int:
		return 0

class DerivedSomething extends Something:
	pass

trait SimSystem:
	func get_value() -> int

func test():
	var s := Something.new()
	var n := NotASystem.new()
	var d := DerivedSomething.new()

	print(s is SimSystem)             # true
	print(n is SimSystem)             # false
	print(d is SimSystem)             # true (inherited from Something)
	print((s as SimSystem) != null)   # true
	print((n as SimSystem) == null)   # true
