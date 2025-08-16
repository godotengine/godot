func test():
	print(E1.V)

enum E1 {V = E2.V}
enum E2 {V = E1.V}
