class A:
	func _init():
		pass

class B extends A: pass
class C extends A: pass

func test():
	var x := B.new()
	print(x is C)
