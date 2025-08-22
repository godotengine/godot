extends 'res://addons/gut/test.gd'

func test_no_asserts():
	pass

func test_passing_assert():
	assert_true(true, 'should pass')

func test_failing_assert():
	assert_true(false, 'should fail')

func test_use_pass_test():
	pass_test('this test passes')

func  test_use_fail_test():
	fail_test('this test fails')

func test_use_pending():
	pending('this is pending')
