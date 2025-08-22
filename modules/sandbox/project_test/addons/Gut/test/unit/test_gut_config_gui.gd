extends GutTest

var GutConfigGui = load('res://addons/gut/gui/gut_config_gui.gd')
var GutConfig = load('res://addons/gut/gut_config.gd')

func _get_default_options():
	var ctrl = add_child_autofree(HBoxContainer.new())
	var gc = GutConfig.new()
	gc.options.double_strategy = GutUtils.get_enum_value(gc.options.double_strategy, GutUtils.DOUBLE_STRATEGY)
	var gcc = GutConfigGui.new(ctrl)
	gcc.set_options(gc.options)
	var opts = gcc.get_options(gc.options)
	return opts


func test_can_make_one():
	var ctrl = add_child_autofree(HBoxContainer.new())
	assert_not_null(autofree(GutConfigGui.new(ctrl)))

func test_free_makes_no_orphans():
	var ctrl = add_child_autofree(HBoxContainer.new())
	var gcc = GutConfigGui.new(ctrl)
	gcc = null
	await wait_physics_frames(1)
	assert_no_new_orphans()

func test_double_strategy_is_script_only():
	var opts = _get_default_options()
	assert_eq(opts.double_strategy, GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY)
