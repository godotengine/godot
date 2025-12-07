class A:
	var _t = 1

	func _priv_method():
		_t = 5

class B:
	var a = A.new()

	func _foo():
		a._t = 2
		a._priv_method()

class C extends A:
	var a = A.new()

	func _foo():
		a._t = 2
		a._priv_method()

func test():
	pass
