class Self:
	var value := "OK"

class TestClass:
	static func make() -> Self:
		return Self.new()

func test():
	var x := TestClass.make()
	prints(x != null, x is Self)
	prints(x.value)
