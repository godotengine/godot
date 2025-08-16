extends GutInternalTester

const TEST_SCRIPT = 'res://test/resources/errors_cause_failures/errors_cause_failures_tests.gd'
var _test_gut = null
var _print_sub_tests = false

func _assert_fail_count_for_test(g, test_name, expected):
	var coll_test = g.get_test_collector().scripts[0].get_test_named(test_name)
	if(coll_test == null):
		fail_test(str(test_name, ' could not be found'))
	else:
		assert_eq(coll_test.fail_texts.size(), expected, str(test_name, ' fail count ', expected))


func before_each():
	_test_gut = add_child_autofree(new_gut(_print_sub_tests))


func test_it():
	_test_gut.add_script(TEST_SCRIPT)
	_test_gut.run_tests()

	var tc = _test_gut.get_test_collector()

	_assert_fail_count_for_test(
		_test_gut, 'test_that_causes_error_passes_when_flag_not_set', 0)
	_assert_fail_count_for_test(
		_test_gut, 'test_that_causes_error_fails_when_flag_set', 1)


