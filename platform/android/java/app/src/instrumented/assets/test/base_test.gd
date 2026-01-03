@abstract class_name BaseTest

var _test_started := 0
var _test_completed := 0
var _test_assert_passes := 0
var _test_assert_failures := 0

@abstract func run_tests()

func __exec_test(test_func: Callable):
	_test_started += 1
	test_func.call()
	_test_completed += 1

func __reset_tests():
	_test_started = 0
	_test_completed = 0
	_test_assert_passes = 0
	_test_assert_failures = 0

func __get_stack_frame():
	for s in get_stack():
		if not s.function.begins_with('__') and s.function != "assert_equal":
			return s
	return null

func __assert_pass():
	_test_assert_passes += 1
	pass

func __assert_fail():
	_test_assert_failures += 1
	var s = __get_stack_frame()
	if s != null:
		print_rich ("[color=red] == FAILURE: In function %s() from '%s' on line %s[/color]" % [s.function, s.source, s.line])
	else:
		print_rich ("[color=red] == FAILURE (run with --debug to get more information!) ==[/color]")

func assert_equal(actual, expected):
	if actual == expected:
		__assert_pass()
	else:
		__assert_fail()
		print ("    |-> Expected '%s' but got '%s'" % [expected, actual])

func assert_true(value):
	if value:
		__assert_pass()
	else:
		__assert_fail()
		print ("    |-> Expected '%s' to be truthy" % value)
