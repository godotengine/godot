const A = preload("out_of_order_external.gd")

class Inner:
	func fn(p1 := A.c1) -> String:
		return "p1=%s" % p1

func f(p := A.c1) -> bool:
	return p is int

enum E2 {V = 2}

enum {EV3 = A.EV1 + 1}
