extends GutTest

var Sandbox_TestsTests = load("res://tests/tests.elf")

func test_math():
	# Create a new sandbox
	var s = Sandbox.new()
	# Set the test program
	s.set_program(Sandbox_TestsTests)

	# 64-bit FP math
	assert_eq(s.vmcall("test_math_sin", 0.0), 0.0)
	assert_eq(s.vmcall("test_math_cos", 0.0), 1.0)
	assert_eq(s.vmcall("test_math_tan", 0.0), 0.0)

	assert_eq(s.vmcall("test_math_asin", 0.0), 0.0)
	assert_eq(s.vmcall("test_math_acos", 1.0), 0.0)
	assert_eq(s.vmcall("test_math_atan", 0.0), 0.0)
	assert_eq(s.vmcall("test_math_atan2", 0.0, 1.0), 0.0)

	assert_eq(s.vmcall("test_math_pow", 2.0, 3.0), 8.0)

	# 32-bit FP math
	assert_eq(s.vmcall("test_math_sinf", 0.0), 0.0)

	assert_eq(s.vmcall("test_math_lerp",       0.0, 1.0, 0.5), 0.5)
	assert_eq(s.vmcall("test_math_smoothstep", 0.0, 1.0, 0.5), 0.5)
	s.queue_free()
