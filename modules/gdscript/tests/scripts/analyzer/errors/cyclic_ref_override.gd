func test():
	print(v)

var v := InnerA.new().f()

class InnerA:
	func f(p := InnerB.new().f()) -> int:
		return 1

class InnerB extends InnerA:
	func f(p := 1) -> int:
		return super.f()
