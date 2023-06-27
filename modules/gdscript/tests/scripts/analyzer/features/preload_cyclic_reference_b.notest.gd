const A = preload("preload_cyclic_reference_a.notest.gd")

static func test_cyclic_reference():
	A.test_cyclic_reference_2()

static func test_cyclic_reference_2():
	A.test_cyclic_reference_3()

static func test_cyclic_reference_3():
	print(A.WAITING_FOR)
