@static_unload

class A:
	static var x: int = 1

	static var y: int = 42:
		set(_value):
			print("The setter is NOT called on initialization.") # GH-77098 p.1

	static func _static_init() -> void:
		prints("A _static_init begin:", x)
		x = -1
		prints("A _static_init end:", x)

	static func sf(p_x: int) -> void:
		x = p_x
		prints("sf:", x)

	# GH-77331
	func f(p_x: int) -> void:
		x = p_x
		prints("f:", x)

class B extends A:
	static func _static_init() -> void:
		prints("B _static_init begin:", x)
		x = -2
		prints("B _static_init end:", x)

	static func sg(p_x: int) -> void:
		x = p_x
		prints("sg:", x)

	func g(p_x: int) -> void:
		x = p_x
		prints("g:", x)

	func h(p_x: int) -> void:
		print("h: call f(%d)" % p_x)
		f(p_x)

func test():
	prints(A.x, B.x)
	A.x = 1 # GH-77098 p.2
	prints(A.x, B.x)
	B.x = 2
	prints(A.x, B.x)

	A.sf(3)
	B.sf(4)
	B.sg(5)

	var b := B.new()
	b.f(6)
	b.g(7)
	b.h(8)
