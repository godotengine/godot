# ------------------------------------------------------------------------------
# This is run by test_errors_cause_failures.gd
# ------------------------------------------------------------------------------
extends GutTest

func before_each():
	gut.treat_error_as_failure = false

func test_that_causes_error_passes_when_flag_not_set():
	get_logger().error('test error')
	pass_test('this is passing')
	assert_true(is_passing(), 'should be passing')

func test_that_causes_error_fails_when_flag_set():
	gut.treat_error_as_failure = true
	get_logger().error('test error')
	pass_test('this is passing')
	assert_true(is_failing(), 'should be failing')
