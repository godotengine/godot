# ##############################################################################
# These classes are used to verify that the befores and afters are being called
# correctly with inner classes.
# ##############################################################################
extends "res://addons/gut/test.gd"

class BeforeAfterCounterTest:
	extends "res://addons/gut/test.gd"

	var before_all_calls = 0
	var before_each_calls = 0
	var after_all_calls = 0
	var after_each_calls = 0

	func before_all():
		before_all_calls += 1

	func before_each():
		before_each_calls += 1

	func after_all():
		after_all_calls += 1

	func after_each():
		after_each_calls += 1


class TestInner1:
	extends BeforeAfterCounterTest

	func test_passing():
		assert_eq(1, 1, '1 = 1')


class TestInner2:
	extends BeforeAfterCounterTest

	func test_passing():
		assert_eq(2, 2, '2 = 2')
