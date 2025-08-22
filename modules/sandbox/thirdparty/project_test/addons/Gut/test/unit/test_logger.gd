extends GutInternalTester

func _new_logger():
	var to_return = GutLogger.new()
	to_return.disable_all_printers(true)
	return to_return

func test_can_warn():
	var l = _new_logger()
	l.warn('something')
	assert_eq(l.get_warnings().size(), 1)

func test_can_error():
	var l = _new_logger()
	l.error('soemthing')
	assert_eq(l.get_errors().size(), 1)

func test_can_info():
	var l = _new_logger()
	l.info('something')
	assert_eq(l.get_infos().size(), 1)

func test_can_debug():
	var l = _new_logger()
	l.debug('something')
	assert_eq(l.get_debugs().size(), 1)

func test_can_deprecate():
	var l = _new_logger()
	l.deprecated('something')
	assert_eq(l.get_deprecated().size(), 1)

func test_clear_clears_all_buffers():
	var l = _new_logger()
	l.debug('a')
	l.info('a')
	l.warn('a')
	l.error('a')
	l.deprecated('a')
	l.clear()
	assert_eq(l.get_debugs().size(), 0, 'debug')
	assert_eq(l.get_infos().size(), 0, 'info')
	assert_eq(l.get_errors().size(), 0, 'error')
	assert_eq(l.get_warnings().size(), 0, 'warnings')
	assert_eq(l.get_deprecated().size(), 0, 'deprecated')

func test_get_set_gut():
	assert_accessors(_new_logger(), 'gut', null, autofree(Gut.new()))

func test_can_get_count_using_type():
	var l = _new_logger()
	l.warn('somethng')
	l.debug('something 2')
	l.debug('something else')
	assert_eq(l.get_count(l.types.debug), 2, 'count debug')
	assert_eq(l.get_count(l.types.warn), 1, 'count warnings')

func test_get_count_with_no_parameter_returns_count_of_all_logs():
	var l = _new_logger()
	l.warn('a')
	l.debug('b')
	l.error('c')
	l.deprecated('d')
	l.info('e')
	assert_eq(l.get_count(), 5)

func test_get_set_indent_level():
	var l = _new_logger()
	assert_accessors(l, 'indent_level', 0, 10)

func test_inc_indent():
	var l = _new_logger()
	l.inc_indent()
	l.inc_indent()
	assert_eq(l.get_indent_level(), 2)

func test_dec_indent_does_not_go_below_0():
	var l = _new_logger()
	l.dec_indent()
	l.dec_indent()
	assert_eq(l.get_indent_level(), 0, 'does not go below 0')

func test_dec_indent_decreases():
	var l = _new_logger()
	l.set_indent_level(10)
	l.dec_indent()
	l.dec_indent()
	l.dec_indent()
	assert_eq(l.get_indent_level(), 7)

func test_get_set_indent_string():
	var l = _new_logger()
	assert_accessors(l, 'indent_string', '    ', "\t")

var log_types = _new_logger().types.keys()
func test_can_enable_disable_types(log_type_key = use_parameters(log_types)):
	var l = _new_logger()
	var log_type = l.types[log_type_key]
	assert_true(l.is_type_enabled(log_type), log_type + ' should be enabled by default')
	l.set_type_enabled(log_type, false)
	assert_false(l.is_type_enabled(log_type), log_type + ' should now be disabled')
