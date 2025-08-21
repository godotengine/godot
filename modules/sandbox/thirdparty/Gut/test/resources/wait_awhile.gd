extends GutTest
# ------------------------------------------------------------------------------
# This is used to give you some time to do stuff before a test script is run.
# This is intended to be used with -gtest option from the command line (hence
# this file does not have the test_ prefix and it is in the resources dir).
# ------------------------------------------------------------------------------
var seconds_to_wait = 10

func test_this_waits_for_a_bit():
	for i in range(seconds_to_wait):
		gut.p(seconds_to_wait - i)
		await wait_seconds(1)
	pass_test("This passes because it just waits")