class A:
	class B:
		func test():
			print(A.B.D)

class C:
	class D:
		pass

func test():
	var inst = A.B.new()
	inst.test()
