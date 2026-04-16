class Outer:
	func wrap() -> Object:
		return null


class Inner:
	func name_of() -> String:
		return "inner"


func test():
	var o = Outer.new():
		func wrap() -> Object:
			return Inner.new():
				func name_of() -> String:
					return "nested"
	print(o.wrap().name_of())
