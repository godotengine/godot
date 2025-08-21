extends 'res://addons/gut/test.gd'

var Gut = load('res://addons/gut/gut.gd')

const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'

func test_can_get_spy():
	var g = autofree(Gut.new())
	assert_ne(g.get_spy(), null)

func test_spy_for_doubler_is_guts_spy():
	var g = autofree(Gut.new())
	assert_eq(g.get_doubler().get_spy(), g.get_spy())


# ---------------------------------
# these two tests use the gut instance that is passed to THIS test.  This isn't
# PURE testing but it appears to cover the bases ok.
class TestGutClearsSpyBetweenTests:
	extends 'res://addons/gut/test.gd'

	func test_spy_cleared_between_tests_setup():
		gut.get_spy().add_call('thing', 'method')
		assert_true(gut.get_spy().was_called('thing', 'method'))

	func test_spy_cleared_between_tests():
		assert_false(gut.get_spy().was_called('thing', 'method'))
# ---------------------------------
