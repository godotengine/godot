extends GutTest


class TestPassingBeforeAllAssertNoOtherTests:
	extends 'res://addons/gut/test.gd'

	func before_all():
		assert_true(true, 'before_all: this should pass')


class TestPassingAfterAllAssertNoOtherTests:
	extends 'res://addons/gut/test.gd'

	func after_all():
		assert_true(true, 'after_all: this should pass')


class TestFailingBeforeAllAssertNoOtherTests:
	extends 'res://addons/gut/test.gd'

	func before_all():
		assert_true(false, 'before_all: this should fail')


class TestFailingAfterAllAssertNoOtherTests:
	extends 'res://addons/gut/test.gd'

	func after_all():
		assert_true(false, 'after_all: this should fail')


class TestHasBeforeAllAfterAllAndSomeTests:
	extends 'res://addons/gut/test.gd'

	func before_all():
		assert_true(true, 'before_all:  should pass')

	func before_each():
		assert_true(true, 'before_each:  should pass')

	func after_each():
		assert_false(true, 'after_each:  should fail')

	func after_all():
		assert_false(true, 'after_all:  should fail')

	func test_this_passes():
		assert_eq(1, 1, 'should pass')

	func test_this_fails():
		fail_test('this fails')


class TestAnotherOneWithSomeTests:
	extends TestHasBeforeAllAfterAllAndSomeTests
