class Outer:
	const OUTER_CONST: = 0
	class Inner:
		pass

func test() -> void:
	var instance: = Outer.Inner.new()
	var instance_v: Variant = instance
	print(instance_v.OUTER_CONST)
