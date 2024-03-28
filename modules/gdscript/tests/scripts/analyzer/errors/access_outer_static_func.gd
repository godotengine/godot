# This works for outer constants, enums, and classes. We _could_ implement this for static functions,
# but it **SHOULD NOT** be done since inner classes are initialized first. See GH-82141.

class InnerClass:
	static func _static_init() -> void:
		outer_static_func()

static func outer_static_func():
	print("This should not be called.")

func test():
	pass
