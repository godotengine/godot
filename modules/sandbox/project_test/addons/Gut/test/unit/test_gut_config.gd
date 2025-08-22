extends GutInternalTester

func _make_gut_config():
	var gc = GutUtils.GutConfig.new()
	gc.logger = GutUtils.GutLogger.new()
	if(gut.log_level < 2):
		gc.logger.disable_all_printers(true)
	return gc


func test_can_make_one():
	var gc = GutUtils.GutConfig.new()
	assert_not_null(gc)


func test_double_strategy_defaults_to_include_native():
	var gc = GutUtils.GutConfig.new()
	assert_eq(gc.default_options.double_strategy, 'SCRIPT_ONLY')


func test_gut_gets_double_strategy_when_applied():
	var gc = GutUtils.GutConfig.new()
	var g = autofree(GutUtils.Gut.new())
	g.log_level = gut.log_level

	gc.options.double_strategy = GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY
	gc.apply_options(g)
	assert_eq(g.double_strategy, gc.options.double_strategy)


func test_gut_gets_default_when_value_invalid():
	var gc = GutUtils.GutConfig.new()
	var g = autofree(GutUtils.Gut.new())
	g.log_level = gut.log_level

	g.double_strategy = GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY
	gc.options.double_strategy = 'invalid value'
	gc.apply_options(g)
	assert_eq(g.double_strategy, GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY)


func test_errors_when_config_file_cannot_be_found():
	var gc = _make_gut_config()
	gc.load_options('res://some_file_that_dne.json')
	assert_errored(gc, 1)


func test_does_not_error_when_default_file_missing():
	var gc = _make_gut_config()
	gc.load_options('res://.gutconfig.json')
	assert_errored(gc, 0)


func test_does_not_error_when_default_editor_file_missing():
	var gc = _make_gut_config()
	gc.load_options(GutUtils.EditorGlobals.editor_run_gut_config_path)
	assert_errored(gc, 0)


func test_errors_when_file_cannot_be_parsed():
	var gc = _make_gut_config()
	gc.load_options('res://addons/gut/gut.gd')
	assert_errored(gc)


func test_errors_when_path_cannot_be_written_to():
	var gc = _make_gut_config()
	gc.write_options("user://some_path/that_does/not_exist/dot.json")
	assert_errored(gc)
