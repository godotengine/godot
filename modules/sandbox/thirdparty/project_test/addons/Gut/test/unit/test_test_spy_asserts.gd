extends GutTest
# ------------------------------------------------------------------------------
# Tests test.gd.  test.gd contains all the asserts and is the class that all
# test scripts inherit from.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class BaseTestClass:
	extends GutInternalTester
	# !! Use this for debugging to see the results of all the subtests that
	# are run using assert_fail_pass, assert_fail and assert_pass that are
	# built into this class
	var _print_all_subtests = false

	# GlobalReset(gr) variables to be used by tests.
	# The values of these are reset in the setup or
	# teardown methods.
	var gr = {
		test = null,
		signal_object = null,
		test_with_gut = null
	}



	# #############
	# Seutp/Teardown
	# #############
	func before_each():
		# Everything in here uses the same logger (the one in `g`) since there
		# should not be any times when they would need to be different and
		# `new_gut` sets up the logger to be more quiet.
		var g = autofree(new_gut(_print_all_subtests))
		g.log_level = 3

		gr.test = Test.new()
		gr.test.set_logger(g.logger)

		gr.test_with_gut = Test.new()
		gr.test_with_gut.gut = g
		gr.test_with_gut.set_logger(g.logger)
		add_child(gr.test_with_gut.gut)

	func after_each():
		gr.test_with_gut.gut.get_spy().clear()

		gr.test.free()
		gr.test = null
		gr.test_with_gut.gut.free()
		gr.test_with_gut.free()


class TestAssertCalled:
	extends BaseTestClass


	func test_passes_when_call_occurred():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.get_value()
		gr.test_with_gut.assert_called(doubled, 'get_value')
		assert_pass(gr.test_with_gut)

	func test_passes_with_parameters():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value(5)
		gr.test_with_gut.assert_called(doubled, 'set_value', [5])
		assert_pass(gr.test_with_gut)

	func test_fails_when_parameters_do_not_match():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value('a')
		gr.test_with_gut.assert_called(doubled, 'set_value', [5])
		assert_fail(gr.test_with_gut)

	func test_works_with_defaults():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_called(doubled, 'has_two_params_one_default', [10, null])
		assert_pass(gr.test_with_gut)

	func test_generates_error_if_third_parameter_not_an_array():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value(5)
		gr.test_with_gut.assert_called(doubled, 'set_value', 5)
		assert_fail(gr.test_with_gut)
		assert_errored(gr.test_with_gut, 1)

	func test_fail_message_indicates_method_does_not_exist():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'foo')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not exist')

	func test_fail_message_indicates_non_included_methods():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'get_parent')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not overload')

	func test_accepts_callable_as_first_parameter():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_called(doubled.has_two_params_one_default)
		assert_pass(gr.test_with_gut)

	func test_when_using_callable_third_parameter_causes_failure():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_called(doubled.has_two_params_one_default, [10, null], ['bad'])
		assert_fail(gr.test_with_gut)

	func test_when_using_callable_second_parameter_causes_failure():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_called(doubled.has_two_params_one_default, [10, null])
		assert_fail(gr.test_with_gut)

	func test_uses_bound_arguments_when_no_parameters_passed_and_there_are_bound_arguments():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_called(doubled.has_two_params_one_default.bind(11, null))
		assert_fail(gr.test_with_gut)





# ------------------------------------------------------------------------------
class TestAssertNotCalled:
	extends BaseTestClass


	func test_passes_when_no_calls_have_been_made():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		gr.test_with_gut.assert_not_called(doubled, 'get_value')
		assert_pass(gr.test_with_gut)

	func test_fails_when_a_call_has_been_made():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.get_value()
		gr.test_with_gut.assert_not_called(doubled, 'get_value')
		assert_fail(gr.test_with_gut)

	func test_fails_when_passed_a_non_doubled_instance():
		gr.test_with_gut.assert_not_called(GDScript.new(), 'method')
		assert_fail(gr.test_with_gut)

	func test_passes_if_parameters_do_not_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(4)
		gr.test_with_gut.assert_not_called(doubled, 'set_value', [5])
		assert_pass(gr.test_with_gut)

	func test_fails_if_parameters_do_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value('a')
		gr.test_with_gut.assert_not_called(doubled, 'set_value', ['a'])
		assert_fail(gr.test_with_gut)

	func test_fails_if_no_params_specified_and_a_call_was_made():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value('a')
		gr.test_with_gut.assert_not_called(doubled, 'set_value')
		assert_fail(gr.test_with_gut)

	func test_fail_message_indicates_method_does_not_exist():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'foo')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not exist')

	func test_fail_message_indicates_non_included_methods():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'get_parent')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not overload')

	func test_accepts_callable_as_first_parameter():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_not_called(doubled.has_two_params_one_default)
		assert_pass(gr.test_with_gut)

	func test_when_using_callable_third_parameter_causes_failure():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_not_called(doubled.has_two_params_one_default, [10, null], ['bad'])
		assert_fail(gr.test_with_gut)

	func test_when_using_callable_second_parameter_causes_failure():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_not_called(doubled.has_two_params_one_default, [10, null])
		assert_fail(gr.test_with_gut)

	func test_uses_bound_arguments_when_no_parameters_passed_and_there_are_bound_arguments():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.has_two_params_one_default(10)
		gr.test_with_gut.assert_not_called(doubled.has_two_params_one_default.bind(10, null))
		assert_fail(gr.test_with_gut)


# ------------------------------------------------------------------------------
class TestAssertCallCount:
	extends BaseTestClass

	func test_assert_call_count_is_deprecated():
		gr.test_with_gut.assert_call_count(self, 'something', 5)
		assert_deprecated(gr.test_with_gut, 1)

	func test_passes_when_nothing_called_and_expected_count_zero():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		gr.test_with_gut.assert_call_count(doubled, 'set_value', 0)
		assert_pass(gr.test_with_gut)

	func test_fails_when_count_does_not_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		gr.test_with_gut.assert_call_count(doubled, 'set_value', 1)
		assert_fail(gr.test_with_gut)

	func test_fails_if_object_is_not_a_double():
		var obj = GDScript.new()
		gr.test_with_gut.gut.get_spy().add_call(obj, '_init')
		gr.test_with_gut.assert_call_count(obj, '_init', 1)
		assert_fail(gr.test_with_gut)

	func test_fails_if_parameters_do_not_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		gr.test_with_gut.assert_call_count(doubled, 'set_value', 2, [5])
		assert_fail(gr.test_with_gut)

	func test_it_passes_if_parameters_do_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		doubled.set_value(5)
		doubled.set_value(5)
		gr.test_with_gut.assert_call_count(doubled, 'set_value', 3, [5])
		assert_pass(gr.test_with_gut)

	func test_when_parameters_not_sent_all_calls_count():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		doubled.set_value(6)
		doubled.set_value(12)
		gr.test_with_gut.assert_call_count(doubled, 'set_value', 4)
		assert_pass(gr.test_with_gut)

	func test_fail_message_indicates_method_does_not_exist():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'foo')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not exist')

	func test_fail_message_indicates_non_included_methods():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called(doubled, 'get_parent')
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not overload')


class TestAssertCalledCount:
	extends BaseTestClass

	func test_passes_when_nothing_called_and_expected_count_zero():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called_count(doubled.set_value, 0)
		assert_pass(gr.test_with_gut)

	func test_fails_when_count_does_not_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		gr.test_with_gut.assert_called_count(doubled.set_value, 1)
		assert_fail(gr.test_with_gut)

	func test_fails_if_object_is_not_a_double():
		var obj = GDScript.new()
		gr.test_with_gut.gut.get_spy().add_call(obj, 'new')
		gr.test_with_gut.assert_called_count(obj.new, 1)
		assert_fail(gr.test_with_gut)

	func test_fails_if_parameters_do_not_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		gr.test_with_gut.assert_called_count(doubled.set_value.bind(5), 2)
		assert_fail(gr.test_with_gut)

	func test_it_passes_if_parameters_do_match():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		doubled.set_value(5)
		doubled.set_value(5)
		gr.test_with_gut.assert_called_count(doubled.set_value.bind(5), 3)
		assert_pass(gr.test_with_gut)

	func test_when_parameters_not_sent_all_calls_count():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		doubled.set_value(10)
		doubled.set_value(6)
		doubled.set_value(12)
		gr.test_with_gut.assert_called_count(doubled.set_value, 4)
		assert_pass(gr.test_with_gut)

	func test_fail_message_indicates_non_included_methods():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		gr.test_with_gut.assert_called_count(doubled.get_parent, 0)
		assert_string_contains(gr.test_with_gut._fail_pass_text[0], 'does not overload')




# ------------------------------------------------------------------------------
class TestGetCallParameters:
	extends BaseTestClass

	func test_it_works():
		var doubled = gr.test_with_gut.double(DoubleMe).new()
		autofree(doubled)
		doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_parameters(doubled, 'set_value'), [5])
		gr.test_with_gut.assert_called(doubled, 'set_value')
		assert_pass(gr.test_with_gut)

	func test_generates_error_if_you_do_not_pass_a_doubled_object():
		var thing = autofree(Node2D.new())
		var _p = gr.test_with_gut.get_call_parameters(thing, 'something')
		assert_errored(gr.test_with_gut, 1)

	func test_accepts_callable_instead_of_object_and_method_name():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_parameters(doubled.set_value), [5])

	func test_gets_indexed_calls_using_callable():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value(5)
		doubled.set_value(6)
		doubled.set_value(7)
		assert_eq(gr.test_with_gut.get_call_parameters(doubled.set_value, 1), [6])

	func test_gets_indexed_calls_using_method_name():
		var doubled = autofree(gr.test_with_gut.double(DoubleMe).new())
		doubled.set_value(5)
		doubled.set_value(6)
		doubled.set_value(7)
		assert_eq(gr.test_with_gut.get_call_parameters(doubled, 'set_value', 1), [6])



# ------------------------------------------------------------------------------
class TestGetCallCount:
	extends BaseTestClass

	func test_it_works():
		var doubled = gr.test_with_gut.partial_double(DoubleMe).new()
		autofree(doubled)
		for i in range(10):
			doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_count(doubled, 'set_value'), 10)

	func test_it_works_with_parameters():
		var doubled = gr.test_with_gut.partial_double(DoubleMe).new()
		autofree(doubled)
		for i in range(3):
			doubled.set_value(3)

		for i in range(5):
			doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_count(doubled, 'set_value', [3]), 3)

	func test_gets_count_when_passed_callable():
		var doubled = gr.test_with_gut.partial_double(DoubleMe).new()
		autofree(doubled)
		for i in range(10):
			doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_count(doubled.set_value), 10)

	func test_it_works_with_parameters_and_callable():
		var doubled = gr.test_with_gut.partial_double(DoubleMe).new()
		autofree(doubled)
		for i in range(3):
			doubled.set_value(3)

		for i in range(5):
			doubled.set_value(5)
		assert_eq(gr.test_with_gut.get_call_count(doubled.set_value.bind(3)), 3)
