extends GutTest

var Gut = load('res://addons/gut/gut.gd')
var ResultExporter = GutUtils.ResultExporter
var GutLogger = GutUtils.GutLogger

var _test_gut = null


# Returns a new gut object, all setup for testing.
func get_a_gut():
	var g = Gut.new()
	g.log_level = g.LOG_LEVEL_ALL_ASSERTS
	g.logger = GutUtils.GutLogger.new()
	g.logger.disable_printer('terminal', true)
	g.logger.disable_printer('gui', true)
	g.logger.disable_printer('console', true)
	return g


func run_scripts(g, one_or_more):
	var scripts = one_or_more
	if(typeof(scripts) != TYPE_ARRAY):
		scripts = [scripts]
	for s in scripts:
		g.add_script(export_script(s))
	g.test_scripts()


func export_script(script_name):
	return str('res://test/resources/exporter_test_files/', script_name)

func before_all():
	GutUtils._test_mode = true

func before_each():
	_test_gut = get_a_gut()
	add_child_autoqfree(_test_gut)


func test_can_make_one():
	assert_not_null(ResultExporter.new())

func test_result_has_testsuites_entry():
	_test_gut.test_scripts()
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	assert_has(result, 'test_scripts')

func test_test_scripts_has_props():
	_test_gut.test_scripts()
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	assert_has(result.test_scripts, 'props')

func test_test_script_props_has_props():
	_test_gut.test_scripts()
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.props
	assert_has(result, 'pending')
	assert_has(result, 'failures')
	assert_has(result, 'tests')
	assert_has(result, 'errors')
	assert_has(result, 'warnings')
	assert_has(result, 'orphans')

func test_test_script_props_have_values_for_one_script():
	run_scripts(_test_gut, 'test_simple.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.props
	assert_eq(result['pending'], 2, 'pending')
	assert_eq(result['failures'], 4, 'failures')
	assert_eq(result['tests'], 8, 'tests')

func test_warnings_and_errors_populated():
	run_scripts(_test_gut, 'test_has_error_and_warning.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.props

	assert_eq(result['errors'], 1, 'errors')
	assert_eq(result['warnings'], 1, 'warnings')

func test_test_scripts_contains_script():
	_test_gut.test_scripts()
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts
	assert_has(result, 'scripts')

func test_scripts_has_script_run():
	run_scripts(_test_gut, 'test_simple.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	assert_has(result, export_script('test_simple.gd'))

func test_script_has_props():
	run_scripts(_test_gut, 'test_simple.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	result = result[export_script('test_simple.gd')]
	assert_has(result, 'props')

func test_script_has_prop_values():
	run_scripts(_test_gut, 'test_simple.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	result = result[export_script('test_simple.gd')]['props']
	assert_has(result, 'tests')
	assert_has(result, 'pending')
	assert_has(result, 'failures')

func test_script_has_proper_prop_values():
	run_scripts(_test_gut, 'test_simple.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	result = result[export_script('test_simple.gd')]['props']
	assert_eq(result['tests'], 8, 'test count')
	assert_eq(result['pending'], 2, 'pending count')
	assert_eq(result['failures'], 4, 'failures')

func test_script_has_proper_prop_values_for_2nd_script():
	run_scripts(_test_gut, ['test_simple.gd', 'test_simple_2.gd'])
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	result = result[export_script('test_simple_2.gd')]['props']
	assert_eq(result['tests'], 3, 'test count')
	assert_eq(result['pending'], 1, 'pending count')
	assert_eq(result['failures'], 1, 'failures')


func test_test_script_props_have_values_for_two_script():
	run_scripts(_test_gut, ['test_simple.gd', 'test_simple_2.gd'])
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.props
	assert_eq(result['pending'], 3, 'pending')
	assert_eq(result['failures'], 5, 'failures')
	assert_eq(result['tests'], 11, 'tests')

func test_totals_with_inner_classes():
	run_scripts(_test_gut, 'test_with_inner_classes.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.props
	assert_eq(result['pending'], 2, 'pending')
	assert_eq(result['failures'], 2, 'failures')
	assert_eq(result['tests'], 6, 'tests')


func test_script_totals_with_inner_classes():
	run_scripts(_test_gut, 'test_with_inner_classes.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts.scripts
	result = result[export_script('test_with_inner_classes.gd.TestClassOne')]['props']
	assert_eq(result['pending'], 1, 'pending')
	assert_eq(result['failures'], 1, 'failures')
	assert_eq(result['tests'], 3, 'tests')

func test_script_has_tests():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut).test_scripts
	result = result.scripts[export_script('test_simple_2.gd')]
	assert_has(result, 'tests')

func test_tests_section_has_tests():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_simple_2.gd')].tests
	assert_has(result, 'test_pass')
	assert_has(result, 'test_fail')
	assert_has(result, 'test_pending')

func test_test_has_status_field():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_simple_2.gd')]
	result = result.tests.test_pass
	assert_has(result, 'status')

func test_test_status_field_has_proper_value():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_simple_2.gd')]
	result = result.tests
	assert_eq(result.test_pass.status, 'pass')
	assert_eq(result.test_fail.status, 'fail')
	assert_eq(result.test_pending.status, 'pending')


func test_test_has_text_fields():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_simple_2.gd')]
	result = result.tests.test_pass
	assert_has(result, 'passing')
	assert_has(result, 'failing')
	assert_has(result, 'pending')

func test_test_has_time_field():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_simple_2.gd')]
	result = result.tests
	assert_has(result.test_pass, 'time_taken')
	assert_has(result.test_fail, 'time_taken')
	assert_has(result.test_pending, 'time_taken')

func test_test_time_taken_in_range():
	run_scripts(_test_gut, 'test_time_taken.gd')
	await wait_for_signal(_test_gut.end_run, 10)
	var re = ResultExporter.new()
	var result = re.get_results_dictionary(_test_gut)
	result = result.test_scripts.scripts[export_script('test_time_taken.gd')]
	assert_has(result, 'tests')
	result = result.tests
	const TIME_ERROR_INTERVAL := 0.1
	assert_almost_eq(result.test_pass_time_taken_about_half_s.time_taken, 0.5, TIME_ERROR_INTERVAL)
	assert_almost_eq(result.test_fail_time_taken_about_half_s.time_taken, 0.5, TIME_ERROR_INTERVAL)
	assert_almost_eq(result.test_pending_time_taken_about_half_s.time_taken, 0.5, TIME_ERROR_INTERVAL)
	assert_almost_eq(result.test_pass_time_taken_about_2s.time_taken, 2.0, TIME_ERROR_INTERVAL)

func test_write_file_creates_file():
	run_scripts(_test_gut, 'test_simple_2.gd')
	var fname = "user://test_result_exporter.json"
	var re = ResultExporter.new()
	var result = re.write_json_file(_test_gut, fname)
	assert_file_not_empty(fname)
	gut.file_delete(fname)

