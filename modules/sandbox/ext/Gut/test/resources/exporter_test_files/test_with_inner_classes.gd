extends 'res://addons/gut/test.gd'


class TestClassOne:
	extends 'res://addons/gut/test.gd'

	func test_pass_1():
		assert_eq('one', 'one')

	func test_fail_1():
		assert_eq(1, 'two')

	func test_pending_with_text():
		pending('this has text')




class TestClassTwo:
	extends 'res://addons/gut/test.gd'

	func test_pass_1():
		assert_eq('one', 'one')

	func test_fail_1():
		assert_eq(1, 'two')

	func test_pending_with_text():
		pending('this has text')

