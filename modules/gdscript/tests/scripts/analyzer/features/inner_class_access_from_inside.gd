# GH-80508

class A:
	func a():
		return A.new()
	func b():
		return B.new()

class B:
	func a():
		return A.new()
	func b():
		return B.new()

func test():
	var a := A.new()
	var b := B.new()
	print(a.a() is A)
	print(a.b() is B)
	print(b.a() is A)
	print(b.b() is B)
