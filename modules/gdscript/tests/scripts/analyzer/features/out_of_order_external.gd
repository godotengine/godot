const B = preload("out_of_order_external_a.notest.gd")

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
	print("B.E2.V: ", B.E2.V)
	print("EV1: ", EV1)
	print("EV2: ", EV2)
	print("B.EV3: ", B.EV3)

var v1 := Inner.new().fn()

class Inner extends B.Inner:
	func fn(p2 := E1.V2) -> String:
		return "%s, p2=%s" % [super.fn(), p2]

var v2 := B.new().f()

const c1 := E1.V1

enum E1 {
	V1 = B.E2.V + 2,
	V2 = V1 - 1
}

const c2 := EV2

enum {
	EV1 = 42,
	EV2 = B.EV3 + 1
}
