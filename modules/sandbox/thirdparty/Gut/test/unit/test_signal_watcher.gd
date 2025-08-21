extends GutInternalTester
# ##############################################################################
# Some notes about signals:
# 	* The parameters appear to be completely optional when defined via
#		add_user_signal.
#	* As long as you have enough parameters defined (defaulted of course) on
#		your handler, then it be notified of the signal when it is emitted.
#	* I seem to remember that if more parameters are passed to the signal
#		handler than the handler has, then the handler will not be called.  This
#		isn't tested here, since it doesn't matter but I thought I would make
#		note of it.
#	*
# ##############################################################################
var SignalWatcher = load('res://addons/gut/signal_watcher.gd')
var gr = {
	so = null,
	sw = null
}

# Constants so I don't get false pass/fail with misspellings
const SIGNALS = {
	NO_PARAMETERS = 'no_parameters',
	ONE_PARAMETER = 'one_parameter',
	TWO_PARAMETERS = 'two_parameters',
	SOME_SIGNAL = 'some_signal'
}

# ####################
# A class that can emit all the signals in SIGNALS
# ####################
class SignalObject:
	extends Node2D

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

# ####################
# Setup/Teardown
# ####################
func before_each():
	gr.sw = autofree(SignalWatcher.new())
	gr.so = autofree(SignalObject.new())

func after_each():
	gr.sw = null
	gr.so = null


# ####################
# Watch one signal
# ####################
func test_no_engine_errors_when_signal_does_not_exist():
	gut.p('!! Look for Red !!')
	gr.sw.watch_signal(gr.so, 'some_signal_that_does_not_exist')
	pass_test('This should generate a GUT warning and not an engine error.')

func test_when_signal_does_not_exist_watch_signal_returns_false():
	var did = gr.sw.watch_signal(gr.so, 'some_signal_that_does_not_exist')
	assert_false(did, 'It should not be watching')

func test_when_signal_does_exist_then_watch_signal_returns_true():
	var did = gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	assert_true(did, 'It should be watching')

# ####################
# Counting Emits
# ####################
func test_when_signal_emitted_the_signal_count_is_incremented():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS)
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.NO_PARAMETERS), 1, 'The signal should have been counted.')

func test_when_signal_emitted_with_parameters_it_is_counted():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS, 'word')
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.NO_PARAMETERS), 1, 'The signal should have been counted.')

func test_when_two_parameter_signal_fired_it_is_counted():
	gr.sw.watch_signal(gr.so, SIGNALS.TWO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.TWO_PARAMETERS)
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.TWO_PARAMETERS), 1)

func test_when_signal_was_not_fired_the_count_is_0():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.SOME_SIGNAL), 0)

func test_when_signal_was_not_being_watched_the_count_is_neg_1():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.SOME_SIGNAL), -1)

func test_when_object_was_not_being_watched_the_count_is_neg_1():
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.SOME_SIGNAL), -1)

func test_can_watch_signals_on_multiple_objects():
	var other_so = autofree(SignalObject.new())
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	gr.sw.watch_signal(other_so, SIGNALS.NO_PARAMETERS)

	gr.so.emit_signal(SIGNALS.SOME_SIGNAL)
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL)

	other_so.emit_signal(SIGNALS.NO_PARAMETERS)
	assert_eq(gr.sw.get_emit_count(gr.so, SIGNALS.SOME_SIGNAL), 2, 'gr.so emit twice')
	assert_eq(gr.sw.get_emit_count(other_so, SIGNALS.NO_PARAMETERS), 1, 'other_so emit once')

# ####################
# did_emit
# ####################
func test_when_signal_was_emitted_did_emit_returns_true():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS)
	assert_true(gr.sw.did_emit(gr.so, SIGNALS.NO_PARAMETERS))

func test_when_signal_was_not_emitted_did_emit_returns_false():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	assert_false(gr.sw.did_emit(gr.so, SIGNALS.NO_PARAMETERS))

func test_when_signal_was_not_being_watched_did_emit_returns_false():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	assert_false(gr.sw.did_emit(gr.so, SIGNALS.SOME_SIGNAL))

func test_when_object_was_not_being_watched_did_emit_returns_false():
	assert_false(gr.sw.did_emit(gr.so, SIGNALS.SOME_SIGNAL))

# ####################
# Signal Parameters
# ####################
func test_can_get_parameters_sent_when_signal_emitted_with_no_parameters():
	gr.sw.watch_signal(gr.so, SIGNALS.NO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS)
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.NO_PARAMETERS)
	assert_eq(params, [], 'It should return an empty array')

func test_when_some_signal_emitted_with_parameter_it_is_returned():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'from_emit')
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.SOME_SIGNAL)
	assert_eq(params, ['from_emit'])

func test_when_two_parameter_signal_fired_the_parameters_are_returned():
	gr.sw.watch_signal(gr.so, SIGNALS.TWO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.TWO_PARAMETERS, 1, 'WORD')
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.TWO_PARAMETERS)
	assert_eq(params, [1, 'WORD'])

func test_get_parameters_returns_null_when_signal_not_fired():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.SOME_SIGNAL)
	assert_eq(params, null)

func test_when_signal_not_watched_null_is_returned():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.NO_PARAMETERS)
	assert_eq(params, null)

func test_when_object_not_watched_null_is_returned():
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.NO_PARAMETERS)
	assert_eq(params, null)

func test_params_returned_default_to_the_most_recent_emission():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'first')
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'second')
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.SOME_SIGNAL)
	assert_eq(params, ['second'])

func test_can_get_params_for_a_specific_emission_of_signal():
	gr.sw.watch_signal(gr.so, SIGNALS.SOME_SIGNAL)
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'first')
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'second')
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL, 'third')
	var params = gr.sw.get_signal_parameters(gr.so, SIGNALS.SOME_SIGNAL, 1)
	assert_eq(params, ['second'])

# ####################
# Watch Script Signals
# ####################
class ScriptSignalObject:
	signal script_signal


func test_can_see_script_signals():
	var script_signaler = ScriptSignalObject.new()
	gr.sw.watch_signals(script_signaler)
	assert_true(gr.sw.is_watching(script_signaler, 'script_signal'))

func test_can_watch_script_signal_explicitly():
	var script_signaler = ScriptSignalObject.new()
	gr.sw.watch_signal(script_signaler, 'script_signal')
	assert_true(gr.sw.is_watching(script_signaler, 'script_signal'))

func test_can_see_script_signals_emit():
	var script_signaler = ScriptSignalObject.new()
	gr.sw.watch_signals(script_signaler)
	script_signaler.emit_signal('script_signal')
	assert_true(gr.sw.did_emit(script_signaler, 'script_signal'))

# ####################
# Watch Signals (plural)
# ####################
func test_watch_signals_watches_all_signals_on_an_object():
	gr.sw.watch_signals(gr.so)
	for sig in SIGNALS:
		assert_true(gr.sw.is_watching(gr.so, SIGNALS[sig]), str('it should be watching: ', SIGNALS[sig]))

func test_watch_signals_ignores_duplicates():
	gr.sw.watch_signals(gr.so)
	gr.sw.watch_signals(gr.so)
	gut.p("-- LOOK FOR RED HERE --")
	assert_true(true)

# ####################
# Clear
# ####################
func test_when_cleared_it_is_not_watching_any_signals():
	gr.sw.watch_signals(gr.so)
	gr.sw.clear()
	for sig in SIGNALS:
		assert_false(gr.sw.is_watching(gr.so, SIGNALS[sig]), str('it should NOT be watching: ', SIGNALS[sig]))

func test_when_cleared_it_should_disconnect_from_signals():
	gr.sw.watch_signals(gr.so)
	gr.sw.clear()
	for sig in SIGNALS:
		assert_false(gr.so.is_connected(SIGNALS[sig],Callable(gr.sw,'_on_watched_signal')), str('it should NOT be connected to: ', SIGNALS[sig]))

func test_clearing_ignores_freed_objecdts():
	add_child(gr.so)
	gr.sw.watch_signals(gr.so)
	gr.so.free()
	await wait_seconds(0.5)
	gr.sw.clear()
	pass_test('we got here')

# ####################
# Get signals emitted
# ####################
func test_when_signal_emitted_it_exists_in_list_of_signals_emitted():
	gr.sw.watch_signals(gr.so)
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS)
	gr.so.emit_signal(SIGNALS.SOME_SIGNAL)
	assert_has(gr.sw.get_signals_emitted(gr.so), SIGNALS.NO_PARAMETERS)
	assert_has(gr.sw.get_signals_emitted(gr.so), SIGNALS.SOME_SIGNAL)
	assert_eq(gr.sw.get_signals_emitted(gr.so).size(), 2)

func test_when_passed_non_watched_obj_emitted_signals_is_empty():
	gr.so.emit_signal(SIGNALS.NO_PARAMETERS)
	assert_eq(gr.sw.get_signals_emitted(gr.so).size(), 0)
