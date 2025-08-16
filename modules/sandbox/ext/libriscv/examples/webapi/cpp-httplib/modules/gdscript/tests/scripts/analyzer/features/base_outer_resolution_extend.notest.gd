extends "base_outer_resolution_base.notest.gd"

const B: = preload("base_outer_resolution_b.notest.gd")

static func test_a(a: A) -> void:
	print(a is A)

static func test_b(b: B) -> void:
	print(b is B)

class InnerClass extends InnerClassInBase:
	static func test_c(c: C) -> void:
		print(c is C)

	class InnerInnerClass:
		static func test_a_b_c(a: A, b: B, c: C) -> void:
			print(a is A and b is B and c is C)

		static func test_enum(test_enum: C.TestEnum) -> void:
			print(test_enum == C.TestEnum.HELLO_WORLD)

		static func test_a_prime(a_prime: A.APrime) -> void:
			print(a_prime is A.APrime)
