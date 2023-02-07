# https://github.com/godotengine/godot/issues/71994

func test():
	pass

class A extends RefCounted:
	pass

class B extends A:
	# Parsing `duplicate()` here would throw this error:
	# Parse Error: The function signature doesn't match the parent. Parent signature is "duplicate(bool = default) -> Resource".
	func duplicate():
		pass
