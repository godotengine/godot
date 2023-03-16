func test():
	print("v1: ", v1)
	print("v1 is String: ", v1 is String)
	print("v2: ", v2)
	print("v2 is bool: ", v2 is bool)
	print("c1: ", c1)
	print("c1 is int: ", c1 is int)
	print("c2: ", c2)
	print("c2 is int: ", c2 is int)
	print("E1.V1: ", E1.V1)
	print("E1.V2: ", E1.V2)
	print("E2.V: ", E2.V)
	print("EV1: ", EV1)
	print("EV2: ", EV2)
	print("EV3: ", EV3)

var v1 := InnerA.new().fn()

class InnerA extends InnerAB:
	func fn(p2 := E1.V2) -> String:
		return "%s, p2=%s" % [super.fn(), p2]

	class InnerAB:
		func fn(p1 := c1) -> String:
			return "p1=%s" % p1

var v2 := f()

func f() -> bool:
	return true

const c1 := E1.V1

enum E1 {
	V1 = E2.V + 2,
	V2 = V1 - 1
}

enum E2 {V = 2}

const c2 := EV2

enum {
	EV1 = 42,
	UNUSED = EV3,
	EV2
}

enum {
	EV3 = EV1 + 1
}
