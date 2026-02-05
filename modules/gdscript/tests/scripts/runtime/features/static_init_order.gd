# GH-82141

class InnerClass:
	static var inner_var: int = _get_value()
	static var inner_inner_instance := InnerInnerClass.new()

	static func _static_init() -> void:
		prints("InnerClass._static_init()", inner_var)

	static func _get_value() -> int:
		prints("InnerClass._get_value()", inner_var)
		return 1

	func _init() -> void:
		prints("InnerClass._init()", inner_var)

	# Out of order: Make sure that the most nested inner class is initialized first,
	# not the first one in the source code.
	class InnerInnerClass:
		static var inner_inner_var: int = _get_value()

		static func _static_init() -> void:
			prints("InnerInnerClass._static_init()", inner_inner_var)

		static func _get_value() -> int:
			prints("InnerInnerClass._get_value()", inner_inner_var)
			return 1

		func _init() -> void:
			prints("InnerInnerClass._init()", inner_inner_var)

static var outer_var: int = _get_value()
static var inner_instance := InnerClass.new()

static func _static_init() -> void:
	prints("_static_init()", outer_var)

static func _get_value() -> int:
	prints("_get_value()", outer_var)
	return 1

func _init() -> void:
	prints("_init()", outer_var)

func test():
	pass
