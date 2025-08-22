# ------------------------------------------------------------------------------
#
# This test script is run manually by test_gut.gd to verify that watched
# signals are cleared between tests.
#
# ------------------------------------------------------------------------------
extends "res://addons/gut/test.gd"

class SignalObject:
	func _init():
		add_user_signal('the_signal')

var the_signal_object = SignalObject.new()

func test_watch_and_emit_a_signal():
	watch_signals(the_signal_object)
	the_signal_object.emit_signal('the_signal')
	assert_signal_emitted(the_signal_object, 'the_signal')

func test_make_sure_not_watching_anymore():
	# this should fail because the object should not be watched anymore,
	# it is verified in the test that runs this script
	assert_signal_emitted(the_signal_object, 'the_signal')
