extends GutTest
# ------------------------------------------------------------------------------
# https://github.com/bitwes/Gut/issues/288
#
# This script ensures that the delay caused by rendering the "waiting...."
# does not affect short yeilds.  It also has a test that can be watched to
# see the "waiting" animation.
#
# after_all performs the assert that makes sure everything is in working order.
#
# The _max_acceptable_time increments are adjusted for running through the
# panel.  The panel takes a lot longer (relatively) than the command line.
#	command line:				3075
#	command line no window:		3072
#	panel:						3248
# ------------------------------------------------------------------------------
var _start_time = 0.0
var _max_acceptable_time = 0.0

func before_all():
	_start_time = Time.get_ticks_msec()

func after_all():
	var total_time = Time.get_ticks_msec() - _start_time
	assert_lt(total_time, _max_acceptable_time)

func test_yield1():
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')

func test_yield2():
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')

func test_yield3():
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')

func test_yield4():
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')

func test_yield5():
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')

var yield_params = [1, 2, 3, 4, 5, 6, 7, 8]
func test_parameterized(p=use_parameters(yield_params)):
	_max_acceptable_time += 20
	await wait_physics_frames(1)
	pass_test('no test')


func test_yield_for_some_seconds_to_watch_animation():
	_max_acceptable_time += 3050
	await wait_seconds(3)
	pass_test('no test')
