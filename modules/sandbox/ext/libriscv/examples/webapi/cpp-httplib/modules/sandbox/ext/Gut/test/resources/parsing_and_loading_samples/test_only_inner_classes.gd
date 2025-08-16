extends "res://addons/gut/test.gd"

class TestInner1:
	extends "res://addons/gut/test.gd"

	func test_passing():
		assert_eq(1, 1, '1 = 1')

	func test_passing2():
		assert_eq(2, 2, '2 = 2')

class TestInner2:
	extends "res://addons/gut/test.gd"

	func test_failing():
		assert_eq(1, 2, '1 != 2')

	func test_failing2():
		assert_eq(2, 3, '2 != 3')
