extends GutInternalTester


var _src_passing_test = """
	func test_is_passing():
		assert_true(true)
	"""
var _src_should_skip_script_method_ret_true = """
	func should_skip_script():
		return true
	"""
var _src_should_skip_script_method_ret_false = """
	func should_skip_script():
		return false
	"""
var _src_should_skip_script_method_ret_string = """
	func should_skip_script():
		return 'skip me'
	"""
var _scr_awaiting_should_skip_script = """
	func should_skip_script():
		print("Awaiting before skipping.")
		await wait_seconds(1)
		print("Now skip.")
		return true
	"""

var _gut = null

func before_all():
	verbose = false
	DynamicGutTest.should_print_source = verbose


func before_each():
	_gut = add_child_autofree(new_gut(verbose))

# --------------
# skip var
# --------------
func test_using_skip_script_variable_is_deprecated():
	var s = DynamicGutTest.new()
	s.add_source("var skip_script = 'skip me thanks'")
	s.add_source(_src_passing_test)
	var t = s.run_test_in_gut(_gut)
	assert_eq(t.deprecated, 1, 'Should be one deprecation.')


func test_when_skip_script_var_is_string_script_is_skipped():
	var s = DynamicGutTest.new()
	s.add_source("var skip_script = 'skip me'")
	s.add_source(_src_passing_test)
	var smry = s.run_test_in_gut(_gut)

	assert_eq(smry.tests, 0, 'no tests should be ran')
	assert_eq(smry.risky, 1, 'Should be marked as risky due to skip')

func test_when_skip_script_var_is_null_the_script_is_ran():
	var s = DynamicGutTest.new()
	s.add_source("var skip_script = null")
	s.add_source(_src_passing_test)

	var smry = s.run_test_in_gut(_gut)
	assert_eq(smry.tests, 1, 'the one test should be ran')
	assert_eq(smry.risky, 0, 'not marked risky just for having var')

func test_when_skip_scrpt_var_is_true_the_script_is_skipped():
	var s = DynamicGutTest.new()
	s.add_source("var skip_script = true")
	s.add_source(_src_passing_test)
	var smry = s.run_test_in_gut(_gut)

	assert_eq(smry.tests, 0, 'no tests should be ran')
	assert_eq(smry.risky, 1, 'Should be marked as risky due to skip')

func test_awaiting_before_should_skip_script():
	var s = DynamicGutTest.new()
	s.add_source(_scr_awaiting_should_skip_script)
	s.run_test_in_gut(_gut)
	await wait_for_signal(_gut.end_run, 3, "Should take exactly 1 second")
	var summery = GutUtils.Summary.new()
	var totals = summery.get_totals(_gut)
	
	assert_eq(totals.tests, 0, 'no tests should be ran')
	assert_eq(totals.risky, 1, 'Should be marked as risky due to skip')


# --------------
# skip method
# --------------
func test_should_skip_script_method_returns_false_by_default():
	var test = autofree(GutTest.new())
	assert_false(test.should_skip_script())


func test_when_should_skip_script_returns_false_script_is_run():
	var s = DynamicGutTest.new()
	s.add_source(_src_should_skip_script_method_ret_false)
	s.add_source(_src_passing_test)

	var smry = s.run_test_in_gut(_gut)
	assert_eq(smry.tests, 1, 'Tests should run')
	assert_eq(smry.risky, 0, 'Should not be risky')


func test_when_should_skip_script_returns_true_script_is_skipped():
	var s = DynamicGutTest.new()
	s.add_source(_src_should_skip_script_method_ret_true)
	s.add_source(_src_passing_test)
	var smry = s.run_test_in_gut(_gut)

	assert_eq(smry.tests, 0, 'no tests should be ran')
	assert_eq(smry.risky, 1, 'Should be marked as risky due to skip')


func test_when_should_skip_script_returns_string_script_is_skipped():
	var s = DynamicGutTest.new()
	s.add_source(_src_should_skip_script_method_ret_string)
	s.add_source(_src_passing_test)
	var smry = s.run_test_in_gut(_gut)

	assert_eq(smry.tests, 0, 'no tests should be ran')
	assert_eq(smry.risky, 1, 'Should be marked as risky due to skip')


func test_using_should_skip_script_method_is_not_deprecated():
	var s = DynamicGutTest.new()
	s.add_source(_src_should_skip_script_method_ret_true)
	s.add_source(_src_passing_test)
	var smry = s.run_test_in_gut(_gut)

	assert_eq(smry.deprecated, 0, 'nothing is deprecated')
