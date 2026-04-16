class Outer:
	class Inner:
		func describe() -> String:
			return "inner"


func test():
	var obj = Outer.Inner.new():
		func describe() -> String:
			return "overridden"
	print(obj.describe())
