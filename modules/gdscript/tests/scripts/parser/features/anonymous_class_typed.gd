class Base:
	func describe() -> String:
		return "base"


func test():
	var obj: Base = Base.new():
		func describe() -> String:
			return "typed"
	print(obj.describe())
