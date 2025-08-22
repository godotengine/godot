extends 'res://addons/gut/test.gd'

class TestYieldInBeforeAll:
	extends 'res://addons/gut/test.gd'

	var value = 'NOT_SET'
	func before_all():
		await wait_seconds(1)
		value = 'set'

	func test_assert_value_set_in_before_all():
		assert_eq(value, 'set')

class TestYieldInAfterAll:
	extends 'res://addons/gut/test.gd'

	var after_all_value = 'NOT_SET'
	func after_all():
		await wait_seconds(1)
		after_all_value = 'set'

	func test_nothing():
		pass_test('this passes')


class TestYieldInAfterEach:
	extends 'res://addons/gut/test.gd'

	var after_each_called = false
	var value = 'NOT_SET'

	func after_each():
		after_each_called = true
		await wait_seconds(1)
		value = 'set'

	# --------------
	# Test order isn't certain so we need two identical tests.
	# --------------
	func test_value_set_in_after_all_1():
		if(after_each_called):
			assert_eq(value, 'set')
		else:
			fail_test('fails, not called yet')

	func test_value_set_in_after_all_2():
		if(after_each_called):
			assert_eq(value, 'set')
		else:
			fail_test('fails, not called yet')


class TestYieldInBeforeEach:
	extends 'res://addons/gut/test.gd'

	var value = 'NOT_SET'

	func before_each():
		await wait_seconds(1)
		value = 'set'

	func test_value_is_set():
		assert_eq(value, 'set')
