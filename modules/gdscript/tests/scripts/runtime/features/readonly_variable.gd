class A:
	@read_only var a: int = 1
	@read_only var b: int
	@read_only var c: int

	func _init(c_val: int) -> void:
		b = 2
		c = c_val

class B extends A:
	@read_only var d: int = 4

	func _init() -> void:
		super._init(12)

func test():
	var a:= A.new(3)
	print("A values")
	print("A.a: ", a.a)
	print("A.b: ", a.b)
	print("A.c: ", a.c)
	print("--")

	var b:= B.new()
	print("B values")
	print("B.a: ", b.a)
	print("B.b: ", b.b)
	print("B.c: ", b.c)
	print("B.d: ", b.d)
	print("--")
