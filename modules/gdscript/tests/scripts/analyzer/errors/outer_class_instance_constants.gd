class Outer:
	const OUTER_CONST: = 0
	class Inner:
		pass

func test() -> void:
	var instance: = Outer.Inner.new()
	print(instance.OUTER_CONST)
