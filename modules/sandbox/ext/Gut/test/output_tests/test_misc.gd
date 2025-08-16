extends GutTest

# Tests taken from test_test_await_methods
class TestTestAwaitMethods:
	extends GutOutputTest

	class PredicateMethods:
		var times_called = 0
		func called_x_times(x):
			times_called += 1
			return times_called == x

	func test_wait_until_accepts_string_as_thrid_arg():
		var pred_methods = PredicateMethods.new()
		var method = pred_methods.called_x_times.bind(10)

		await wait_until(method, 1.1, 'DID YOU SEE THIS?')
		look_for("DID YOU SEE THIS?")

	func test_wait_until_accepts_time_between_then_msg():
		var pred_methods = PredicateMethods.new()
		var method = pred_methods.called_x_times.bind(10)

		await wait_until(method, 1.1, .25, 'DID YOU SEE THIS?')
		look_for('DID YOU SEE THIS?')
		assert_eq(pred_methods.times_called, 4)


class TestResultExporter:
	extends GutInternalTester

	var _test_gut = null

	func get_a_gut():
		var g = Gut.new()
		g.log_level = g.LOG_LEVEL_ALL_ASSERTS
		g.logger = GutUtils.GutLogger.new()
		g.logger.disable_printer('terminal', true)
		g.logger.disable_printer('gui', true)
		g.logger.disable_printer('console', true)
		return g

	func export_script(script_name):
		return str('res://test/resources/exporter_test_files/', script_name)


	func run_scripts(g, one_or_more):
		var scripts = one_or_more
		if(typeof(scripts) != TYPE_ARRAY):
			scripts = [scripts]
		for s in scripts:
			g.add_script(export_script(s))
		g.test_scripts()

	func before_each():
		_test_gut = get_a_gut()
		add_child_autoqfree(_test_gut)


	func test_spot_check():
		run_scripts(_test_gut, ['test_simple_2.gd', 'test_simple.gd', 'test_with_inner_classes.gd'])
		var re = GutUtils.ResultExporter.new()
		var result = re.get_results_dictionary(_test_gut)
		GutUtils.pretty_print(result)
		pass_test("Check output")


class TestTest:
	extends GutOutputTest

	var _gut = null
	var _test = null

	func before_each():
		_gut = new_gut()
		add_child_autofree(_gut)
		_gut.log_level = 4

		_test = new_wired_test(_gut)

	func test_children_warning():
		var TestClass = load('res://addons/gut/test.gd')
		for i in range(3):
			var extra_test = TestClass.new()
			add_child(extra_test)
		should_warn("There should be warnings")


	func test_fails_with_message_if_non_doubled_passed():
		var obj = GDScript.new()
		_test.gut.get_spy().add_call(obj, 'method')
		_test.assert_called(obj, 'method1')
		should_error("non doubled error")#gut.p('!! Check output !!')
		assert_fail(_test)


class TestSummary:
	extends GutOutputTest

	var Summary = load('res://addons/gut/test_collector.gd')
	const PARSING_AND_LOADING = 'res://test/resources/parsing_and_loading_samples'
	const SUMMARY_SCRIPTS = 'res://test/resources/summary_test_scripts'

	var _gut = null

	func before_all():
		verbose = true # should always print everything

	func before_each():
		_gut = new_gut()
		add_child_autofree(_gut)
		_gut._lgr._min_indent_level = 5

		_gut.logger.disable_printer("terminal", false)
		_gut._should_print_summary = true



	func _run_test_gut_tests(test_gut):
		test_gut.p(" ------------------ start test output ------------------")
		watch_signals(test_gut)
		test_gut.run_tests()
		if(get_signal_emit_count(test_gut, 'end_run') == 0):
			await wait_for_signal(test_gut.end_run, 60, 'waiting for tests to finish')
		test_gut.p(" ------------------ end test output ------------------")

		gut.p("\n\n\n\n\n\n\n")


	func test_output_1():
		_gut.add_directory(PARSING_AND_LOADING)

		await _run_test_gut_tests(_gut)
		just_look_at_it("Look at the output, or don't if you aren't interested.")

	func test_output_with_unit_and_script_set():
		_gut.add_directory(PARSING_AND_LOADING)
		_gut.select_script('sample')
		_gut.unit_test_name = 'number'

		await _run_test_gut_tests(_gut)
		just_look_at_it("Look at the output, or don't if you aren't interested.")

	func test_output_with_scripts_that_have_issues():
		_gut.add_directory(SUMMARY_SCRIPTS)
		_gut.log_level = 99
		_gut.select_script('issues')

		await _run_test_gut_tests(_gut)
		just_look_at_it("Look at the output, or don't if you aren't interested.")

	func test_output_with_risky_tests():
		_gut.add_directory(SUMMARY_SCRIPTS)
		_gut.log_level = 99
		_gut.select_script('risky_and_passing')

		await _run_test_gut_tests(_gut)
		just_look_at_it("Look at the output, or don't if you aren't interested.")

	func test_output_with_all_test_scripts():
		_gut.add_directory(SUMMARY_SCRIPTS)
		_gut.add_directory(PARSING_AND_LOADING)

		_gut.log_level = 99
		await _run_test_gut_tests(_gut)
		just_look_at_it("Look at the output, or don't if you aren't interested.")


class TestJunitXmlExport:
	extends GutOutputTest

	var _gut = null

	func run_scripts(g, one_or_more):
		var scripts = one_or_more
		if(typeof(scripts) != TYPE_ARRAY):
			scripts = [scripts]
		for s in scripts:
			g.add_script(export_script(s))
		g.test_scripts()

	func export_script(fname):
		return str('res://test/resources/exporter_test_files/', fname)

	func before_all():
		verbose = true # should always print everything

	func before_each():
		_gut = new_gut()
		add_child_autofree(_gut)

	func test_spot_check():
		run_scripts(_gut, ['test_simple_2.gd', 'test_simple.gd', 'test_with_inner_classes.gd'])
		var re = GutUtils.JunitXmlExport.new()
		var result = re.get_results_xml(_gut)
		print(result)
		just_look_at_it('Check Output')


class TestDiffTool:
	extends GutOutputTest

	var DiffTool = GutUtils.DiffTool

	func test_summarize():
		var d1 = {'aa':'asdf', 'a':1, 'b':'two', 'c':autofree(Node2D.new())}
		var d2 = {'a':1.0, 'b':2, 'c':GutUtils.Strutils.new(), 'cc':'adsf'}
		var dd = DiffTool.new(d1, d2)
		gut.p(dd.summarize())
		just_look_at_it('Visually check this')

	func test_with_obj_as_keys():
		var d1 = {}
		var d2 = {}
		var node_1 = autofree(Node2D.new())
		var node_2 = autofree(Node2D.new())
		var other_1 = autofree(GutUtils.Strutils.new())
		var other_2 = autofree(GutUtils.Strutils.new())
		for i in range(6):
			var key = autofree(GutUtils.Strutils.new())

			if(i%2 == 0):
				d1[key] = node_1
				d2[key] = node_2
			else:
				d1[key] = other_1
				d2[key] = other_2

		var dd =  DiffTool.new(d1, d2)
		gut.p(dd.summarize())
		just_look_at_it('Visually check this')
