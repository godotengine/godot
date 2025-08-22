# This script is used by test_test.gd/TestTestStateChecking.
# Each of these tests are run by that test class and the
# pass/fail count is checked.
extends 'res://addons/gut/test.gd'

class TestIsPassing:
	extends 'res://addons/gut/test.gd'

	func test_is_passing_returns_true_when_test_is_passing():
		assert_true(true, 'should pass')
		assert_true(is_passing(), 'should pass')

	func test_is_passing_returns_false_when_test_is_failing():
		assert_true(false, 'should fail')
		assert_false(is_passing(), 'should pass')

	func test_is_passing_false_by_default():
		assert_false(is_passing(), 'this should pass')

	func test_is_passing_returns_true_before_test_fails():
		assert_true(true, 'should pass')
		assert_true(is_passing(), 'should pass')
		assert_true(false, 'should fail')

class TestIsFailing:
	extends 'res://addons/gut/test.gd'

	func test_is_failing_returns_true_when_failing():
		assert_false(true, 'should fail')
		assert_true(is_failing(), 'should pass')

	func test_is_failing_returns_false_when_passing():
		assert_true(true, 'should pass')
		assert_false(is_failing(), 'should pass')

	func test_is_failing_returns_false_by_default():
		assert_false(is_failing())

	func test_is_failing_returns_false_before_test_passes():
		assert_false(is_failing(), 'should pass')
		assert_true(true, 'should pass')

class TestUseIsPassingInBeforeAll:
	extends 'res://addons/gut/test.gd'

	func before_all():
		var is_it = is_passing()

	func test_nothing():
		pass_test('pass it')

class TestUseIsPassingInAfterAll:
	extends 'res://addons/gut/test.gd'

	func after_all():
		var is_it = is_passing()

	func test_nothing():
		pass_test('pass it')


class TestUseIsFailingInBeforeAll:
	extends 'res://addons/gut/test.gd'

	func before_all():
		var is_it = is_failing()

	func test_nothing():
		pass_test('pass it')

class TestUseIsFailingInAfterAll:
	extends 'res://addons/gut/test.gd'

	func after_all():
		var is_it = is_failing()

	func test_nothing():
		pass_test('pass it')
