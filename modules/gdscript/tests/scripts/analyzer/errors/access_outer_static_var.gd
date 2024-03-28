# This works for outer constants, enums, and classes. We _could_ implement this for static variables,
# but it **SHOULD NOT** be done since inner classes are initialized first. See GH-82141.

static var outer_static_var: int

class InnerClass:
	static func _static_init() -> void:
		print(outer_static_var)

func test():
	pass
