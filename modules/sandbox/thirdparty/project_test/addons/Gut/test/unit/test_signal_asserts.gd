extends GutTest

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



# ------------------------------------------------------------------------------
class TestSignalAsserts:
	extends BaseTestClass

	# Constants for all the signals created in SignalObject so I don't get false
	# pass/fail from typos
	const SIGNALS = {
		NO_PARAMETERS = 'no_parameters',
		ONE_PARAMETER = 'one_parameter',
		TWO_PARAMETERS = 'two_parameters',
		SOME_SIGNAL = 'some_signal',
		SCRIPT_SIGNAL = 'script_signal'
	}

	# ####################
	# A class that can emit all the signals in SIGNALS
	# ####################
	class SignalObject:
		signal script_signal

		func _init():
			add_user_signal(SIGNALS.NO_PARAMETERS)
			add_user_signal(SIGNALS.ONE_PARAMETER, [
				{'name':'something', 'type':TYPE_INT}
			])
			add_user_signal(SIGNALS.TWO_PARAMETERS, [
				{'name':'num', 'type':TYPE_INT},
				{'name':'letters', 'type':TYPE_STRING}
			])
			add_user_signal(SIGNALS.SOME_SIGNAL)

	func before_each():
		super.before_each()
		gr.signal_object = SignalObject.new()

	func after_each():
		super.after_each()
		gr.signal_object = null

	func test_when_object_not_being_watched__assert_signal_emitted__fails():
		gr.test.assert_signal_emitted(gr.signal_object, SIGNALS.SOME_SIGNAL)
		assert_fail(gr.test)

	func test_when_signal_emitted__assert_signal_emitted__passes():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emitted(gr.signal_object, SIGNALS.SOME_SIGNAL)
		assert_pass(gr.test)

	func test_when_signal_not_emitted__assert_signal_emitted__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted(gr.signal_object, SIGNALS.SOME_SIGNAL)
		assert_fail(gr.test)

	func test_when_object_does_not_have_signal__assert_signal_emitted__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted(gr.signal_object, 'signal_does_not_exist')
		assert_fail(gr.test, 1, 'Only the failure that it does not have signal should fire.')

	func test_when_signal_emitted__assert_signal_not_emitted__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_not_emitted(gr.signal_object, SIGNALS.SOME_SIGNAL)
		assert_fail(gr.test)

	func test_when_signal_not_emitted__assert_signal_not_emitted__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_not_emitted(gr.signal_object, SIGNALS.SOME_SIGNAL)
		assert_pass(gr.test)

	func test_when_object_does_not_have_signal__assert_signal_not_emitted__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_not_emitted(gr.signal_object, 'signal_does_not_exist')
		assert_fail(gr.test, 1, 'Only the failure that it does not have signal should fire.')

	func test_when_signal_emitted_once__assert_signal_emit_count__passes_with_1():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emit_count(gr.signal_object, SIGNALS.SOME_SIGNAL, 1)
		assert_pass(gr.test)

	func test_when_signal_emitted_twice__assert_signal_emit_count__fails_with_1():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emit_count(gr.signal_object, SIGNALS.SOME_SIGNAL, 1)
		assert_fail(gr.test)

	func test_when_object_does_not_have_signal__assert_signal_emit_count__fails():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emit_count(gr.signal_object, 'signal_does_not_exist', 0)
		assert_fail(gr.test)

	func test_assert_signal_emit_count_accepts_signal_instead_of_name():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit()
		gr.test.assert_signal_emit_count(gr.signal_object.script_signal, 1)
		assert_pass(gr.test)

	func test_assert_signal_emit_count_uses_text_when_signal_passed_instead_of_name():
		gr.test.watch_signals(gr.signal_object)
		# gr.signal_object.script_signal.emit()
		gr.test.assert_signal_emit_count(gr.signal_object.script_signal, 1, '__foo__')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, '__foo__')

	func test__assert_has_signal__passes_when_it_has_the_signal():
		gr.test.assert_has_signal(gr.signal_object, SIGNALS.NO_PARAMETERS)
		assert_pass(gr.test)

	func test__assert_has_signal__fails_when_it_does_not_have_the_signal():
		gr.test.assert_has_signal(gr.signal_object, 'signal does not exist')
		assert_fail(gr.test)

	func test_can_get_signal_emit_counts():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		assert_eq(gr.test.get_signal_emit_count(gr.signal_object, SIGNALS.SOME_SIGNAL), 2)

	func test_text_included_in_output_when_signal_name_passed():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted(gr.signal_object, 'script_signal', '__foo__')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, '__foo__')

	func test_accepts_signal_object_instead_of_name():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit()
		gr.test.assert_signal_emitted(gr.signal_object.script_signal)
		assert_pass(gr.test)

	func test_text_included_in_output_when_signal_passed():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted(gr.signal_object.script_signal, '__foo__')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, '__foo__')

	func test_not_text_included_in_output_when_signal_name_passed():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit()
		gr.test.assert_signal_not_emitted(gr.signal_object, 'script_signal', '__foo__')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, '__foo__')

	func test_not_accepts_signal_object_instead_of_name():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_not_emitted(gr.signal_object.script_signal)
		assert_pass(gr.test)

	func test_not_text_included_in_output_when_signal_passed():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_not_emitted(gr.signal_object.script_signal, '__foo__')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, '__foo__')

	func test_text_is_empty_string_when_only_signal_object_is_passed():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted(gr.signal_object, 'script_signal')
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, ']:   (')


	# ------------------
	# With Parameters
	func test__with_parameters_errors_when_parameters_are_not_an_array():
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, 1)
		assert_errored(gr.test)
		assert_fail(gr.test)

	func test__assert_signal_emitted_with_parameters__fails_when_object_not_watched():
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [])
		assert_fail(gr.test)

	func test__assert_signal_emitted_with_parameters__passes_when_parameters_match():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [1])
		assert_pass(gr.test)


	func test__assert_signal_emitted_with_parameters__passes_when_all_parameters_match():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1, 2, 3)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [1, 2, 3])
		assert_pass(gr.test)

	func test__assert_signal_emitted_with_parameters__fails_when_signal_not_emitted():
		gr.test.watch_signals(gr.signal_object)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [2])
		assert_fail(gr.test)

	func test__assert_signal_emitted_with_parameters__fails_when_parameters_dont_match():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [2])
		assert_fail(gr.test)

	func test__assert_signal_emitted_with_parameters__fails_when_not_all_parameters_match():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1, 2, 3)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [1, 0, 3])
		assert_fail(gr.test)

	func test__assert_signal_emitted_with_parameters__can_check_multiple_emission():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 2)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [1], 0)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [2], 1)
		assert_pass(gr.test, 2)

	func test_when_signal_emit_with_parameters_fails_because_signal_was_not_emitted_then_signals_are_listed():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.NO_PARAMETERS)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SCRIPT_SIGNAL, [0])
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, SIGNALS.NO_PARAMETERS)
		assert_string_contains(text, SIGNALS.SOME_SIGNAL)

	func test_issue_152():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1.0, 2, 3.0)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, [1, 2.0, 3])
		assert_fail(gr.test)

	func test_accepts_signal_instead_of_signal_name():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit(1, 2, 3)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object.script_signal, [1, 2, 3])
		assert_pass(gr.test)

	func test_accepts_signal_instead_of_signal_name_and_index():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit(1, 2, 3)
		gr.signal_object.script_signal.emit(4, 5, 6)
		gr.test.assert_signal_emitted_with_parameters(gr.signal_object.script_signal, [1, 2, 3], 0)
		assert_pass(gr.test)


	func test_can_get_signal_parameters():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL, 1, 2, 3)
		assert_eq(gr.test.get_signal_parameters(gr.signal_object, SIGNALS.SOME_SIGNAL, 0), [1, 2, 3])

	func test_can_get_signal_parameters_using_signal_object_instead_of_name():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit(1, 2, 3)
		assert_eq(gr.test.get_signal_parameters(gr.signal_object.script_signal, 0), [1, 2, 3])

	func test_can_get_signal_parameters_using_signal_object_instead_of_name_without_index():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit(1, 2, 3)
		assert_eq(gr.test.get_signal_parameters(gr.signal_object.script_signal), [1, 2, 3])

	func test_can_get_signal_parameters_using_signal_object_instead_of_name_with_different_index():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.script_signal.emit(1, 2, 3)
		gr.signal_object.script_signal.emit(4, 5, 6)
		gr.signal_object.script_signal.emit(7, 8, 9)
		assert_eq(gr.test.get_signal_parameters(gr.signal_object.script_signal, 1), [4, 5, 6])


	func test__assert_signal_emitted__passes_with_script_signals():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.SCRIPT_SIGNAL)
		gr.test.assert_signal_emitted(gr.signal_object, SIGNALS.SCRIPT_SIGNAL)
		assert_pass(gr.test)

	func test__assert_has_signal__works_with_script_signals():
		gr.test.assert_has_signal(gr.signal_object, SIGNALS.SCRIPT_SIGNAL)
		assert_pass(gr.test)

	func test_when_signal_emitted_fails_emitted_signals_are_listed():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.NO_PARAMETERS)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emitted(gr.signal_object, SIGNALS.SCRIPT_SIGNAL)
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, SIGNALS.NO_PARAMETERS)
		assert_string_contains(text, SIGNALS.SOME_SIGNAL)

	func test_when_signal_count_fails_then_emitted_signals_are_listed():
		gr.test.watch_signals(gr.signal_object)
		gr.signal_object.emit_signal(SIGNALS.NO_PARAMETERS)
		gr.signal_object.emit_signal(SIGNALS.SCRIPT_SIGNAL)
		gr.signal_object.emit_signal(SIGNALS.SOME_SIGNAL)
		gr.test.assert_signal_emit_count(gr.signal_object, SIGNALS.SCRIPT_SIGNAL, 2)
		var text = gr.test._fail_pass_text[0]
		assert_string_contains(text, SIGNALS.NO_PARAMETERS)
		assert_string_contains(text, SIGNALS.SOME_SIGNAL)

	func test__get_signal_emit_count__returns_neg_1_when_not_watched():
		assert_eq(gr.test.get_signal_emit_count(gr.signal_object, SIGNALS.SOME_SIGNAL), -1)

	func test_get_signal_emit_count_works_with_signal_ref():
		assert_eq(gr.test.get_signal_emit_count(gr.signal_object.script_signal), -1)




# ------------------------------------------------------------------------------
class TestConnectionAsserts:
	extends BaseTestClass

	const SIGNAL_NAME = 'test_signal'
	const METHOD_NAME = 'test_signal_connector'

	class Signaler:
		signal test_signal

	class ConnectTo:
		func test_signal_connector():
			pass

	func test_when_target_connected_to_source_connected_passes_with_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.connect(SIGNAL_NAME,Callable(c,METHOD_NAME))
		gr.test.assert_connected(s, c, SIGNAL_NAME, METHOD_NAME)
		assert_pass(gr.test)

	func test_when_target_connected_to_source_connected_passes_without_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.connect(SIGNAL_NAME,Callable(c,METHOD_NAME))
		gr.test.assert_connected(s, c, SIGNAL_NAME)
		assert_pass(gr.test)

	func test_when_target_not_connected_to_source_connected_fails_with_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		gr.test.assert_connected(s, c, SIGNAL_NAME, METHOD_NAME)
		assert_fail(gr.test)

	func test_when_target_not_connected_to_source_connected_fails_without_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		gr.test.assert_connected(s, c, SIGNAL_NAME)
		assert_fail(gr.test)

	func test_when_target_connected_to_source_not_connected_fails_with_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.connect(SIGNAL_NAME,Callable(c,METHOD_NAME))
		gr.test.assert_not_connected(s, c, SIGNAL_NAME, METHOD_NAME)
		assert_fail(gr.test)

	func test_when_target_connected_to_source_not_connected_fails_without_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.connect(SIGNAL_NAME,Callable(c,METHOD_NAME))
		gr.test.assert_not_connected(s, c, SIGNAL_NAME)
		assert_fail(gr.test)

	func test_when_target_not_connected_to_source_not_connected_passes_with_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		gr.test.assert_not_connected(s, c, SIGNAL_NAME, METHOD_NAME)
		assert_pass(gr.test)

	func test_when_target_not_connected_to_source_not_connected_passes_without_method_name():
		var s = Signaler.new()
		var c = ConnectTo.new()
		gr.test.assert_not_connected(s, c, SIGNAL_NAME)
		assert_pass(gr.test)

	func test_assert_conneccted_accepts_objects_instead_of_names():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.test_signal.connect(c.test_signal_connector)
		gr.test.assert_connected(s.test_signal, c)
		assert_pass(gr.test)

	func test_assert_conneccted_works_with_callable():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.test_signal.connect(c.test_signal_connector)
		gr.test.assert_connected(s.test_signal, c.test_signal_connector)
		assert_pass(gr.test)

	func test_assert_not_connected_accepts_objects_instead_of_names():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.test_signal.connect(c.test_signal_connector)
		gr.test.assert_not_connected(s.test_signal, c)
		assert_fail(gr.test)

	func test_assert_not_conneccted_works_with_callable():
		var s = Signaler.new()
		var c = ConnectTo.new()
		s.test_signal.connect(c.test_signal_connector)
		gr.test.assert_not_connected(s.test_signal, c.test_signal_connector)
		assert_fail(gr.test)

