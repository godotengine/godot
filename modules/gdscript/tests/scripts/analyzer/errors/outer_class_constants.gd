class Outer:
	const OUTER_CONST: = 0
	class Inner:
		pass

func test() -> void:
	var type: = Outer.Inner
	print(type.OUTER_CONST)
