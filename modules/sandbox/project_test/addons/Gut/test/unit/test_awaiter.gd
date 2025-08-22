extends GutTest

var Awaiter = load('res://addons/gut/awaiter.gd')

class Counter:
	extends Node

	var idle_time = 0.0
	var idle_frames = 0

	var physics_time = 0.0
	var physics_frames = 0

	func _process(delta):
		idle_time += delta
		idle_frames += 1

	func _physics_process(delta):
		physics_time += delta
		physics_frames += 1

	func stop_counting_on_signal(the_signal):
		the_signal.connect(func():
			set_process(false)
			set_physics_process(false))


class Signaler:
	signal the_signal
	signal with_parameters(foo, bar)


class PredicateMethods:
	var times_called = 0

	func called_x_times(x):
		times_called += 1
		return times_called == x

	func not_called_x_times(x):
		times_called += 1
		return times_called != x




func test_is_not_paused_by_default():
	var a = add_child_autofree(Awaiter.new())
	assert_false(a.is_waiting())

func test_did_last_wait_timeout_is_false_by_default():
	var a = add_child_autoqfree(Awaiter.new())
	assert_false(a.did_last_wait_timeout)

func test_did_last_wait_timeout_is_readonly():
	var a = add_child_autoqfree(Awaiter.new())
	a.did_last_wait_timeout = true
	assert_false(a.did_last_wait_timeout)

func test_wait_started_emitted_when_waiting_seconds():
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_seconds(.5)
	assert_signal_emitted(a, 'wait_started')

func test_signal_emitted_after_half_second():
	# important that counter added to tree before awaiter.  If it is after, then
	# the last _process call for the counter will happen after the signal in
	# the awaiter is sent and the counts are off.
	var c = add_child_autoqfree(Counter.new())
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_seconds(.5)
	await a.timeout
	assert_signal_emitted(a, 'timeout')
	assert_gt(c.physics_time, .49, 'waited enough time')

func test_is_waiting_while_waiting_on_time():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_seconds(.5)
	await get_tree().create_timer(.1).timeout
	assert_true(a.is_waiting())

func test_wait_for_sets_did_last_wait_timeout_to_true():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_seconds(.2)
	await a.timeout
	assert_true(a.did_last_wait_timeout)

func test_wait_for_resets_did_last_wait_timeout():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_seconds(.2)
	await a.timeout
	a.wait_seconds(20)
	assert_false(a.did_last_wait_timeout)

func test_wait_process_frames_counts_frames_in_process(_x = run_x_times(10)):
	var a = add_child_autoqfree(Awaiter.new())
	var c = add_child_autoqfree(Counter.new())
	watch_signals(a)
	a.wait_process_frames(10)
	await a.timeout
	# TODO figure this out:
	# When the whole script is ran, the first time in here it appears to take
	# 11 frames.  All subsequent runs are 10 frames.  If you run just this
	# test, then all are 10 frames.  Smells funny, would like to know more.
	if(_x == 0):
		assert_between(c.idle_frames, 10, 11)
	else:
		assert_eq(c.idle_frames, 10)

func test_wait_phyiscs_frames_counts_frames_in_physics_process(_x = run_x_times(10)):
	var a = add_child_autoqfree(Awaiter.new())
	var c = add_child_autoqfree(Counter.new())
	watch_signals(a)
	a.wait_physics_frames(10)
	await a.timeout
	assert_eq(c.physics_frames, 10, 'waited enough frames')

func test_wait_started_emitted_when_waiting_frames():
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_physics_frames(10)
	assert_signal_emitted(a, 'wait_started')

func test_signal_emitted_after_10_frames():
	var c = add_child_autoqfree(Counter.new())
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_physics_frames(10)
	await a.timeout
	assert_signal_emitted(a, 'timeout')
	assert_eq(c.physics_frames, 10, 'waited enough frames')

func test_is_waiting_while_waiting_on_frames():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_physics_frames(120)
	await get_tree().create_timer(.1).timeout
	assert_true(a.is_waiting())

func test_wait_physics_frames_sets_did_last_wait_timeout_to_true():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_physics_frames(10)
	await a.timeout
	assert_true(a.did_last_wait_timeout)

func test_wait_process_frames_sets_did_last_wait_timeout_to_true():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_process_frames(10)
	await a.timeout
	assert_true(a.did_last_wait_timeout)

func test_wait_physics_frames_resets_did_last_wait_timeout():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_physics_frames(10)
	await a.timeout
	a.wait_physics_frames(50)
	assert_false(a.did_last_wait_timeout)

func test_wait_process_frames_resets_did_last_wait_timeout():
	var a = add_child_autoqfree(Awaiter.new())
	a.wait_process_frames(10)
	await a.timeout
	a.wait_process_frames(50)
	assert_false(a.did_last_wait_timeout)

func test_wait_started_emitted_when_waiting_on_signal():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, 10)
	assert_signal_emitted(a, 'wait_started')

func test_can_wait_for_signal():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, 10)
	await get_tree().create_timer(.5).timeout
	s.the_signal.emit()
	# gotta wait for the 2 additional frames
	await get_tree().create_timer(.05).timeout
	assert_signal_emitted(a, 'timeout')

func test_can_wait_for_signal_with_parameters():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.with_parameters, 10)
	await get_tree().create_timer(.5).timeout
	s.with_parameters.emit(1, 2)
	# gotta wait for the 2 additional frames
	await get_tree().create_timer(.05).timeout
	assert_signal_emitted(a, 'timeout')

func test_after_wait_for_signal_signal_is_disconnected():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, 10)
	await get_tree().create_timer(.5).timeout
	s.the_signal.emit()
	assert_not_connected(s, a, 'the_signal')

func test_when_signal_not_emitted_max_time_is_waited():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, .5)
	await get_tree().create_timer(.8).timeout
	assert_signal_emitted(a, 'timeout')

func test_is_waiting_when_waiting_on_signal():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, .5)
	await get_tree().create_timer(.1).timeout
	assert_true(a.is_waiting())

func test_is_not_paused_when_signal_emitted_before_max_time():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, 10)
	await get_tree().create_timer(.5).timeout
	s.the_signal.emit()
	# gotta wait for the 2 additional frames
	await get_tree().create_timer(.05).timeout
	assert_false(a.is_waiting())

func test_after_timeout_signal_is_disconnected():
	var s = Signaler.new()
	var a = add_child_autoqfree(Awaiter.new())
	watch_signals(a)
	a.wait_for_signal(s.the_signal, .1)
	await get_tree().create_timer(.5).timeout
	assert_not_connected(s, a, 'the_signal')

func test_wait_for_signal_sets_did_last_wait_timeout_to_true_when_time_exceeded():
	var a = add_child_autoqfree(Awaiter.new())
	var s = Signaler.new()

	a.wait_for_signal(s.the_signal, .5)
	await a.timeout

	assert_true(a.did_last_wait_timeout)

func test_wait_for_signal_resets_did_last_wait_timeout_when_signal_detected():
	var a = add_child_autoqfree(Awaiter.new())

	a.wait_seconds(.2)
	await a.timeout # results in true timeout

	var s = Signaler.new()
	a.wait_for_signal(s.the_signal, 10)
	await get_tree().create_timer(.1).timeout
	# Checking for did_last_wait_timeout while it is still waiting
	# on the signal to be emitted.
	assert_false(a.did_last_wait_timeout)

func test_wait_for_signal_did_last_time_out_false_when_does_not_timeout():
	var a = add_child_autoqfree(Awaiter.new())
	var s = Signaler.new()

	a.wait_for_signal(s.the_signal, 10)
	await get_tree().create_timer(.5).timeout
	s.the_signal.emit()
	await get_tree().create_timer(.5).timeout
	assert_false(a.is_waiting(), 'is_waiting')
	assert_false(a.did_last_wait_timeout, 'timed_out')




class TestWaitUntil:
	extends GutTest
	var Awaiter = load('res://addons/gut/awaiter.gd')

	func test_wait_until_emits_wait_started():
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_until(func(): return true, 10)

		assert_signal_emitted(a, 'wait_started')

	func test_wait_until_ignores_ints_values():
		var a = add_child_autoqfree(Awaiter.new())
		a.wait_until(func(): return 1, .25)
		await a.timeout
		assert_true(a.did_last_wait_timeout)

	func test_wait_until_ignores_strings_values():
		var a = add_child_autoqfree(Awaiter.new())
		a.wait_until(func(): return 'true', .25)
		await a.timeout
		assert_true(a.did_last_wait_timeout)

	func test_wait_until_ignores_object_values():
		var a = add_child_autoqfree(Awaiter.new())
		a.wait_until(func(): return self, .25)
		await a.timeout
		assert_true(a.did_last_wait_timeout)

	func test_wait_until_waits_until_predicate_function_is_true():
		var node = add_child_autoqfree(Node.new())
		var is_named_foo = func(): return node.name == 'foo'
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_until(is_named_foo, 10)
		await get_tree().create_timer(.5).timeout
		node.name = 'foo'

		await get_tree().create_timer(.05).timeout
		assert_signal_emitted(a, 'timeout')
		assert_false(a.did_last_wait_timeout)


	func test_wait_until_reaches_timeout_when_predicate_function_never_returns_true():
		var never = func(): return false
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_until(never, .5)
		await get_tree().create_timer(.8).timeout

		assert_signal_emitted(a, 'timeout')
		assert_true(a.did_last_wait_timeout)

	func test_wait_until_causes_is_waiting_to_be_true_when_waiting():
		var never = func(): return false
		var a = add_child_autoqfree(Awaiter.new())

		a.wait_until(never, .5)
		await get_tree().create_timer(.1).timeout

		assert_true(a.is_waiting())

	func test_wait_until_causes_is_waiting_to_be_false_when_predicate_function_returns_true_before_timeout():
		var node = add_child_autoqfree(Node.new())
		var is_named_foo = func(): return node.name == 'foo'
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_until(is_named_foo, 10)
		await get_tree().create_timer(.5).timeout
		node.name = 'foo'

		# gotta wait for the 2 additional frames
		await get_tree().create_timer(.05).timeout
		assert_false(a.is_waiting())


	func test_wait_until_uses_time_between_calls():
		var pred_methods = PredicateMethods.new()
		var method = pred_methods.called_x_times.bind(10)
		var a = add_child_autoqfree(Awaiter.new())

		a.wait_until(method, 1.1, .25)
		await a.timeout
		assert_eq(pred_methods.times_called, 4)




class TestWaitWhile:
	extends GutTest
	var Awaiter = load('res://addons/gut/awaiter.gd')

	func test_wait_while_emits_wait_started():
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_while(func(): return false, 10)

		assert_signal_emitted(a, 'wait_started')


	func test_wait_while_waits_until_predicate_function_is_false():
		var node = add_child_autoqfree(Node.new())
		var is_not_named_foo = func(): return node.name != 'foo'
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_while(is_not_named_foo, 10)
		await get_tree().create_timer(.5).timeout
		node.name = 'foo'

		await get_tree().create_timer(.05).timeout
		assert_signal_emitted(a, 'timeout')
		assert_false(a.did_last_wait_timeout)


	func test_wait_while_reaches_timeout_when_predicate_function_never_returns_false():
		var never_false = func(): return true
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_while(never_false, .5)
		await get_tree().create_timer(.8).timeout

		assert_signal_emitted(a, 'timeout')
		assert_true(a.did_last_wait_timeout)

	func test_wait_while_causes_is_waiting_to_be_true_when_waiting():
		var never_false = func(): return true
		var a = add_child_autoqfree(Awaiter.new())

		a.wait_while(never_false, .5)
		await get_tree().create_timer(.1).timeout

		assert_true(a.is_waiting())

	func test_wait_while_causes_is_waiting_to_be_false_when_predicate_function_returns_true_before_timeout():
		var node = add_child_autoqfree(Node.new())
		var is_named_foo = func(): return node.name == 'foo'
		var a = add_child_autoqfree(Awaiter.new())
		watch_signals(a)

		a.wait_while(is_named_foo, 10)
		await get_tree().create_timer(.5).timeout
		node.name = 'foo'

		# gotta wait for the 2 additional frames
		await get_tree().create_timer(.05).timeout
		assert_false(a.is_waiting())


	func test_wait_while_uses_time_between_calls():
		var pred_methods = PredicateMethods.new()
		var method = pred_methods.not_called_x_times.bind(10)
		var a = add_child_autoqfree(Awaiter.new())

		a.wait_while(method, 1.1, .25)
		await a.timeout
		assert_eq(pred_methods.times_called, 4)

	var some_var
	func test_wait_ends_when_non_boolean_value_returned():
		var a = add_child_autoqfree(Awaiter.new())
		var method = func():
			if(str(some_var) == '1'):
				return true
			else:
				return some_var

		some_var == 1
		a.wait_while(method, 1.1, .25)
		await get_tree().create_timer(.5).timeout

		some_var = 'hello world'
		await get_tree().create_timer(.5).timeout
		assert_false(a.is_waiting())


