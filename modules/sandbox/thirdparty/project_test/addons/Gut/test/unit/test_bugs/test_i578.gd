extends GutTest

class InputSingletonTracker:
	extends Node
	var pressed_frames = []
	var just_pressed_count = 0
	var just_released_count = 0

	var _frame_counter = 0

	func _process(_delta):
		_frame_counter += 1

		if(Input.is_action_just_pressed("jump")):
			just_pressed_count += 1

		if(Input.is_action_just_released("jump")):
			just_released_count += 1

		if Input.is_action_pressed("jump"):
			pressed_frames.append(_frame_counter)



# ------------------------------------------------------------------------------
# There are a few tests in here that will fail if the window changes to
# a different monitor at any point before these tests are run.  I was able to
# replicate the issue consistently.
#
# Fails when (MacOS)
# * Drag the window to a different monitor.
# * Use keystroke (better-snap-tools) to move window.
# * Window is dragged and "held" on another monitor.
#
# Passes when (MacOS)
# * The window is not moved.
# * Window is moved around on the same monitor.
# * The window is being "held" on the original monitor.
# * Window is moved back to original monitor before these tests are executed.
#   This appears to be the case regardless of the number of times the window
#   changes monitor.
#
# To test these scenarios I used the following to run a script that just delays
# for a bit, and then run this script.
# gdscript addons/gut/gut_cmdln.gd -gexit -gconfig= -gtest test/resources/wait_awhile.gd,test/unit/test_bugs/test_i578.gd
# ------------------------------------------------------------------------------
class TestInputSingleton:
	extends "res://addons/gut/test.gd"
	var _sender = InputSender.new(Input)

	func before_all():
		_sender.release_all()
		_sender.clear()
		await wait_physics_frames(10)
		InputMap.add_action("jump")


	func after_all():
		InputMap.erase_action("jump")


	func after_each():
		_sender.release_all()
		# Wait for key release to be processed. Otherwise the key release is
		# leaked to the next test and it detects an extra key release.
		await wait_physics_frames(1)
		_sender.clear()


	func test_raw_input_press():
		var r = add_child_autofree(InputSingletonTracker.new())

		Input.action_press("jump")
		await wait_physics_frames(2)
		Input.action_release("jump")

		# see inner-test-class note
		assert_gt(r.pressed_frames.size(), 1, 'input size (FAILS if window changes monitor)')

	func test_input_sender_press():
		var r = add_child_autofree(InputSingletonTracker.new())

		_sender.action_down("jump").hold_for('10f')
		await wait_for_signal(_sender.idle, 5)

		assert_gt(r.pressed_frames.size(), 1, 'input size')

	func test_input_sender_just_pressed():
		var r = add_child_autofree(InputSingletonTracker.new())

		_sender.action_down("jump").hold_for("20f")
		await wait_physics_frames(5)

		assert_eq(r.just_pressed_count, 1, 'just pressed once')
		assert_eq(r.just_released_count, 0, 'not released yet')

	func test_input_sender_just_released():
		var r = add_child_autofree(InputSingletonTracker.new())

		_sender.action_down("jump").hold_for('5f')
		await wait_for_signal(_sender.idle, 10)

		assert_eq(r.just_pressed_count, 1, 'just pressed once')
		# see inner-test-class note
		assert_eq(r.just_released_count, 1, 'released key once (FAILS if window changes monitor)')
