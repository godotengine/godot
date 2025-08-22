extends GutTest

class CoupledScriptTest:
	extends GutTest

	var _gut = null

	func before_each():
		_gut = GutUtils.Gut.new()
		_gut._should_print_versions = false
		_gut._should_print_summary = false
		add_child_autofree(_gut)

	func _same_name():
		return gut.get_current_test_object().name

	func _run_tests(script_path, inner_class, test_name):
		_gut.add_script(script_path)

		if(inner_class != null):
			_gut.inner_class_name = inner_class

		if(test_name != null):
			_gut.unit_test_name = test_name

		_gut.test_scripts()


	func _get_inner_class_script_instance(inner_class):
		var inst = null
		var scripts = _gut._test_script_objects
		var idx = 0
		while(inst == null and idx < scripts.size()):
			if(_str(scripts[idx]).ends_with(inner_class + ')')):
				inst = scripts[idx]
			else:
				idx += 1
		return inst

	func _assert_pass_fail_count(passing, failing):
		assert_eq(_gut.get_pass_count(), passing, 'Pass count does not match')
		assert_eq(_gut.get_fail_count(), failing, 'Failing count does not match')


class TestYieldInBeforeAfterMethods:
	extends CoupledScriptTest

	const SCRIPT_PATH = 'res://test/resources/yield_in_before_after_methods.gd'

	func test_gut_waits_for_yield_in_before_all():
		_run_tests(SCRIPT_PATH, 'TestYieldInBeforeAll', null)
		await wait_for_signal(_gut.end_run, 10)
		_assert_pass_fail_count(1, 0)

	func test_gut_waits_for_yield_in_after_all():
		_run_tests(SCRIPT_PATH, 'TestYieldInAfterAll', null)
		await wait_for_signal(_gut.end_run, 10)
		var test_script = _get_inner_class_script_instance('TestYieldInAfterAll')
		assert_eq(test_script.after_all_value, 'set')

	func test_gut_waits_for_yield_in_after_each():
		_run_tests(SCRIPT_PATH, 'TestYieldInAfterEach', null)
		await wait_for_signal(_gut.end_run, 10)
		_assert_pass_fail_count(1, 1)

	func test_gut_waits_for_yield_in_before_each():
		_run_tests(SCRIPT_PATH, 'TestYieldInBeforeEach', null)
		await wait_for_signal(_gut.end_run, 10)
		_assert_pass_fail_count(1, 0)

	# func test_all_with_non_gut_yield_methods():
	# 	_run_tests(SCRIPT_PATH, 'TestYieldsThatDoNotUseGutYieldMethods', null)
	# 	await yield_to(_gut, 'end_run', 10)
	# 	var test_script = _get_inner_class_script_instance('TestYieldsThatDoNotUseGutYieldMethods')
	# 	assert_eq(test_script.before_all_value, 'set', 'before_all_value')
	# 	assert_eq(test_script.before_each_value, 'set', 'before_each_value')

class TestOutputWhenRunDirectly:
	extends GutTest

	var before_all_value = 'NOT_SET'
	var before_each_value = 'NOT_SET'
	var after_each_value = 'NOT_SET'
	var after_all_value = 'NOT_SET'

	var _timer = Timer.new()

	func before_all():
		gut.p('WATCH THESE TESTS')
		add_child(_timer)
		_timer.set_wait_time(3)
		_timer.one_shot = true

		_timer.start()
		gut.p('before_all yield')
		await _timer.timeout
		before_all_value = 'set'

	func before_each():
		_timer.start()
		gut.p('before_each yield')
		await _timer.timeout
		before_each_value = 'set'

	func after_each():
		_timer.start()
		gut.p('after_each yield')
		await _timer.timeout
		after_each_value = 'set'

	func after_all():
		_timer.start()
		gut.p('after_all yield')
		await _timer.timeout
		# must be queued free b/c after yield _timer is still emitting
		# the timeout signal.  This results in a false positive message
		# indicating there are still children in the test.
		_timer.queue_free()

	func test_one():
		assert_eq(before_all_value, 'set', 'before all value')
		assert_eq(before_each_value, 'set', 'before each value')
		_timer.start()
		gut.p('test_one yield')
		await _timer.timeout


