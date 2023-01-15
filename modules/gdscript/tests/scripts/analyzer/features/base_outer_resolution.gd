const A: = preload("base_outer_resolution_a.notest.gd")
const B: = preload("base_outer_resolution_b.notest.gd")
const C: = preload("base_outer_resolution_c.notest.gd")

const Extend: = preload("base_outer_resolution_extend.notest.gd")

func test() -> void:
	Extend.test_a(A.new())
	Extend.test_b(B.new())
	Extend.InnerClass.test_c(C.new())
	Extend.InnerClass.InnerInnerClass.test_a_b_c(A.new(), B.new(), C.new())
	Extend.InnerClass.InnerInnerClass.test_enum(C.TestEnum.HELLO_WORLD)
	Extend.InnerClass.InnerInnerClass.test_a_prime(A.APrime.new())
