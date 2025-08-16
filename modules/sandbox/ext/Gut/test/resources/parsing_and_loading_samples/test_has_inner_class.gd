extends "res://addons/gut/test.gd"
func before_all():
	gut.p('script:  pre-run')
func before_each():
	gut.p('script:  setup')
func after_each():
	gut.p('script:  teardown')
func after_all():
	gut.p('script:  post-run')

func test_something():
	assert_true(true)

func test_nothing():
	assert_true(false)

class TestClass1:
	extends "res://addons/gut/test.gd"
	func before_all():
		gut.p('TestClass1:  pre-run')
	func before_each():
		gut.p('TestClass1:  setup')
	func after_each():
		gut.p('TestClass1:  teardown')
	func after_all():
		gut.p('TestClass1:  post-run')

	func test_context1_one():
		assert_true(true)

	func test_context1_two():
		pending()

	func test_failing():
		assert_eq(2, 1)

	func print_something():
		print('hello world')

class NotTestClass:
	func test_something():
		pass
	func not_a_test():
		pass
