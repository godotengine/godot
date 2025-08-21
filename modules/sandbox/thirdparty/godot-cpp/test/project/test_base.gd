extends Node

var test_passes := 0
var test_failures := 0

func __get_stack_frame():
	var me = get_script()
	for s in get_stack():
		if s.source == me.resource_path:
			return s
	return null

func __assert_pass():
	test_passes += 1

func __assert_fail():
	test_failures += 1
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

func assert_true(v):
	assert_equal(v, true)

func assert_false(v):
	assert_equal(v, false)

func assert_not_equal(actual, expected):
	if actual != expected:
		__assert_pass()
	else:
		__assert_fail()
		print ("    |-> Expected '%s' NOT to equal '%s'" % [expected, actual])

func exit_with_status() -> void:
	var success: bool = (test_failures == 0)
	print ("")
	print_rich ("[color=%s] ==== TESTS FINISHED ==== [/color]" % ("green" if success else "red"))
	print ("")
	print_rich ("   PASSES: [color=green]%s[/color]" % test_passes)
	print_rich ("   FAILURES: [color=red]%s[/color]" % test_failures)
	print ("")

	if success:
		print_rich("[color=green] ******** PASSED ******** [/color]")
	else:
		print_rich("[color=red] ******** FAILED ********[/color]")
	print("")

	get_tree().quit(0 if success else 1)
