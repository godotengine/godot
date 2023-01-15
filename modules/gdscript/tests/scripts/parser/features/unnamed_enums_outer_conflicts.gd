class A:
	enum { X = 1 }

	class B:
		enum { X = 2 }

class C:
	const X = 3

	class D:
		enum { X = 4 }

func test():
	print(A.X)
	print(A.B.X)
	print(C.X)
	print(C.D.X)
