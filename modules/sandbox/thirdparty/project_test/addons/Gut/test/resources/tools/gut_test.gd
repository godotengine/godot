class_name GutInternalTester
extends GutTest

var verbose = false

const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
var DoubleMe = GutUtils.WarningsManager.load_script_ignoring_all_warnings(DOUBLE_ME_PATH)

const DOUBLE_ME_SCENE_PATH = 'res://test/resources/doubler_test_objects/double_me_scene.tscn'
var DoubleMeScene = GutUtils.WarningsManager.load_script_ignoring_all_warnings(DOUBLE_ME_SCENE_PATH)

const DOUBLE_EXTENDS_NODE2D = 'res://test/resources/doubler_test_objects/double_extends_node2d.gd'
var DoubleExtendsNode2D = GutUtils.WarningsManager.load_script_ignoring_all_warnings(DOUBLE_EXTENDS_NODE2D)

const DOUBLE_EXTENDS_WINDOW_DIALOG = 'res://test/resources/doubler_test_objects/double_extends_window_dialog.gd'
var DoubleExtendsWindowDialog = GutUtils.WarningsManager.load_script_ignoring_all_warnings(DOUBLE_EXTENDS_WINDOW_DIALOG)

const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
var InnerClasses = GutUtils.WarningsManager.load_script_ignoring_all_warnings(INNER_CLASSES_PATH)

var Gut = load('res://addons/gut/gut.gd')
var Test = load('res://addons/gut/test.gd')
var GutLogger = load('res://addons/gut/logger.gd')
var Spy = load('res://addons/gut/spy.gd')
var TestCollector = load('res://addons/gut/test_collector.gd')


func _init():
	GutUtils._test_mode = true


func _get_logger_from_obj(obj):
	var to_return = null
	if(obj.has_method('get_logger')):
		to_return = obj.get_logger()
	elif(obj.get('logger') != null):
		to_return = obj.logger
	return to_return


func _assert_log_count(entries, type, count):
	if(count == -1):
		assert_gt(entries.size(), 0, str('There should be at least 1 ' + type))
	else:
		assert_eq(entries.size(), count, str('There should be ', count, ' ', type))


const SHOULD_PASS = &"Should pass"
const SHOULD_FAIL = &"Should fail"

func print_fail_pass_text(t):
	for i in range(t._fail_pass_text.size()):
		gut.p('sub-test:  ' + t._fail_pass_text[i], gut.LOG_LEVEL_FAIL_ONLY)


func assert_warn(obj, times=1):
	var lgr = _get_logger_from_obj(obj)
	if(lgr != null):
		_assert_log_count(lgr.get_warnings(), 'warnings', times)
	else:
		_fail(str('Cannot assert_errored, ', obj, ' does not have get_logger method or logger property'))


func assert_errored(obj, times=1):
	var things_lgr = _get_logger_from_obj(obj)
	if(things_lgr != null):
		_assert_log_count(things_lgr.get_errors(), 'errors', times)
	else:
		_fail(str('Cannot assert_errored, ', obj, ' does not have get_logger method or logger property'))


func assert_deprecated(obj, times=1):
	var lgr = _get_logger_from_obj(obj)
	if(lgr != null):
		_assert_log_count(lgr.get_deprecated(), 'deprecated', times)
	else:
		_fail(str('Cannot assert_errored, ', obj, ' does not have get_logger method or logger property'))


func assert_has_logger(obj):
	assert_has_method(obj, 'get_logger')
	assert_has_method(obj, 'set_logger')
	if(obj.has_method('get_logger')):
		assert_not_null(obj.get_logger(), 'Default logger not null.')
		if(obj.has_method('set_logger')):
			var l = double(GutLogger).new()
			obj.set_logger(l)
			assert_eq(obj.get_logger(), l, 'Set/get works')


func assert_fail_pass(t, fail_count, pass_count, msg=''):
	var self_fail_count = get_fail_count()
	assert_eq(t.get_fail_count(), fail_count, 'Bad FAIL COUNT:  ' + msg)
	assert_eq(t.get_pass_count(), pass_count, 'Bad PASS COUNT:  ' + msg)
	if(get_fail_count() != self_fail_count or verbose):
		print_fail_pass_text(t)

# convenience method to assert the number of failures on the gr.test_gut object.
func assert_fail(t, count=1, msg=''):
	var self_fail_count = get_fail_count()
	assert_eq(t.get_fail_count(), count, 'Expected FAIL COUNT:  ' + msg)
	if(t.get_pass_count() > 0 and count != t.get_assert_count()):
		assert_eq(t.get_pass_count(), 0, 'When checking for failures there should be no passing')
	if(get_fail_count() != self_fail_count or verbose):
		print_fail_pass_text(t)

# convenience method to assert the number of passes on the gr.test_gut object.
func assert_pass(t, count=1, msg=''):
	var self_fail_count = get_fail_count()
	assert_eq(t.get_pass_count(), count, 'Expected PASS COUNT:  ' + msg)
	if(t.get_fail_count() != 0 and count != t.get_assert_count()):
		assert_eq(t.get_fail_count(), 0, 'When checking for passes there should be no failures.')
	if(get_fail_count() != self_fail_count or verbose):
		print_fail_pass_text(t)

func assert_fail_msg_contains(t, text):
	if(t.get_fail_count() != 1):
		assert_fail(t, 1, 'assert_fail_msg_contains requires single failing assert.')
	elif(t.get_pass_count() != 0):
		assert_pass(t, 0, 'assert_fail_msg_contains requires no passing asserts.')
	else:
		assert_string_contains(t._fail_pass_text[0], text)

func get_error_count(obj):
	return obj.logger.get_errors().size()

var new_gut_indent_string = "|   "
func new_gut(print_sub_tests=false):
	var g = Gut.new()
	g.logger = GutLogger.new()
	g.logger.disable_all_printers(true)
	g.update_loggers()

	g.log_level = 3
	if(print_sub_tests):
		g.logger.disable_printer("terminal", false)
		g.logger._min_indent_level = 1
		g.logger.dec_indent()
		g.logger.set_indent_string(new_gut_indent_string)
		g.logger.disable_formatting(!print_sub_tests)
		g.logger.set_type_enabled(g.logger.types.debug, true)

	g._should_print_versions = false
	g._should_print_summary = false

	return g


func new_partial_double_gut(print_sub_tests=false):
	var g = partial_double(Gut).new()
	g.logger = GutUtils.GutLogger.new()
	g.logger.disable_all_printers(true)
	g.update_loggers()

	if(print_sub_tests):
		g.log_level = 3
		g.logger.disable_printer("terminal", false)
		g.logger._min_indent_level = 1
		g.logger.dec_indent()
		g.logger.set_indent_string(new_gut_indent_string)
		g.logger.disable_formatting(!print_sub_tests)
	else:
		g.log_level = g.LOG_LEVEL_FAIL_ONLY

	g._should_print_versions = false
	g._should_print_summary = false

	return g


func new_no_print_logger(override=!verbose):
	var to_return = GutLogger.new()
	to_return.disable_all_printers(override)
	return to_return


func new_wired_test(gut_instance):
	var t = GutTest.new()
	t.gut = gut_instance
	t.set_logger(gut_instance.logger)
	return t

# ----------------------------
# Not used yet, but will be used eventually

# func new_test_double():
# 	var t = double(GutTest).new()
# 	var logger = double(GutUtils.GutLogger).new()
# 	stub(t, 'set_logger').to_call_super()
# 	stub(t, 'get_logger').to_call_super()
# 	t.set_logger(logger)
# 	return t
# ----------------------------

