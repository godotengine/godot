class Outer:
	const OUTER_CONST: = 0
	class Inner:
		pass

func test() -> void:
	var type: = Outer.Inner
	var type_v: Variant = type
	print(type_v.OUTER_CONST)
