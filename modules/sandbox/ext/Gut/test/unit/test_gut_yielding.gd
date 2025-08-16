extends GutTest
#-------------------------------------------------------------------------------
# All of these tests require some amount of user interaction or verifying of the
# output.
#-------------------------------------------------------------------------------
class TestPauseBeforeTeardown:
	extends "res://addons/gut/test.gd"
	var timer = null

	func before_all():
		timer = Timer.new()
		timer.set_one_shot(true)
		add_child(timer)

	func after_all():
		timer.free()

	func before_each():
		timer.set_wait_time(1)

	func test_wait_for_continue_click():
		gut.p('should have had to press continue')
		gut.pause_before_teardown()
		pass_test('Got here')

	func test_can_pause_twice():
		gut.p('should have had to press continue')
		gut.pause_before_teardown()
		pass_test('Got here')

	func test_can_pause_after_yielding():
		pass_test('should have seen a pause and press continue')
		gut.p('yielding for 1 second')
		timer.start()
		await timer.timeout
		gut.p('done yielding')
		gut.pause_before_teardown()

	func test_can_call_pause_before_yielding():
		pass_test('should  see a pause')
		gut.pause_before_teardown()
		gut.p('yielding for 1 second')
		timer.start()
		await timer.timeout
		gut.p('done yielding')

	func test_can_pause_between_each_parameterized_test(p=use_parameters([1, 2, 3])):
		assert_between(p, -10, 10)
		pause_before_teardown()
