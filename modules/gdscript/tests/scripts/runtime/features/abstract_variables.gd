@abstract class A:
	@abstract var text_1: String
	@abstract var text_2: String

class B extends A:
	var text_1 = "text_1b"
	var text_2 = "text_2b"

class C extends B:
	var text_1 = "text_1c"

@abstract class D extends A:
	var text_1 = "text_1d"

class E extends D:
	var text_2 = "text_2e"

func test():
	var b:= B.new()
	print("B text_1= " + b.text_1)
	print("B text_2= " + b.text_2)

	var c := C.new()
	print("C text_1= " + c.text_1)
	print("C text_2= " + c.text_2)

	var e := E.new()
	print("E text_1= " + e.text_1)
	print("E text_2= " + e.text_2)

	e.text_1 = "text_1e"
	print("E text_1= " + e.text_1)
