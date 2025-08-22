extends GutTest

# test.gd has a StubParams variable already so this has to have a
# different name.  I thought it was too vague to just use the one
# that test.gd has
const STUB_PARAMS_PATH = 'res://addons/gut/stub_params.gd'
var StubParamsClass = load('res://addons/gut/stub_params.gd')

func find_method_meta(methods, method_name):
	var meta = null
	var idx = 0
	while (idx < methods.size() and meta == null):
		var m = methods[idx]
		if(m.name == method_name):
			meta = m
		idx += 1

	return meta

var gr = {
	stub_params = null
}

func before_each():
	gr.stub_params = StubParamsClass.new()


func test_can_make_one():
	assert_not_null(StubParamsClass.new())


func test_to_return_sets_return_value():
	gr.stub_params.to_return(7)
	assert_eq(gr.stub_params.return_val, 7)

func test_to_return_returns_self():
	var thing = gr.stub_params.to_return(7)
	assert_eq(thing, gr.stub_params)

func test_init_sets_stub_target():
	var s = StubParamsClass.new('thing')
	assert_eq(s.stub_target, 'thing')

func test_init_sets_method():
	var s = StubParamsClass.new('thing', 'method')
	assert_eq(s.stub_method, 'method')

func test_when_passed_returns_self():
	var thing = gr.stub_params.when_passed(7)
	assert_eq(thing, gr.stub_params)

func test_when_passed_sets_parameters():
	gr.stub_params.when_passed(1)
	assert_eq(gr.stub_params.parameters, [1])

func test_parameters_turn_values_into_array():
	gr.stub_params.when_passed(1,2,3)
	assert_eq(gr.stub_params.parameters, [1,2,3])

func test_can_take_up_to_10_parameters():
	gr.stub_params.when_passed(1,2,3,4,5,6,7,8,9,10)
	assert_eq(gr.stub_params.parameters.size(), 10)

func test_can_set_to_call_super():
	gr.stub_params.to_call_super()
	assert_eq(gr.stub_params.call_super, true)

func test_to_call_super_returns_self():
	var val = gr.stub_params.to_call_super()
	assert_eq(val, gr.stub_params)

func test_to_do_nothing_returns_self():
	var sp = StubParamsClass.new('thing', 'method')
	assert_eq(sp.to_do_nothing(), sp)

func test_turns_paths_into_classes():
	var sp = StubParamsClass.new(STUB_PARAMS_PATH, 'to_return')
	assert_eq(sp.stub_target, StubParamsClass)

func test_to_call_is_null_by_default():
	var sp = StubParamsClass.new(self.assert_true)
	assert_null(sp.call_this)

func test_to_call_sets_call_this():
	var sp = StubParamsClass.new(self.assert_true)
	sp.to_call(self.assert_false)
	assert_eq(sp.call_this, self.assert_false)

func test_to_call_returns_itself():
	var sp = StubParamsClass.new(self.assert_true)
	var val = sp.to_call(self.assert_false)
	assert_eq(val, sp)


# --------------
# Parameter Count and Defaults
# --------------
func test_param_count_returns_self():
	var val = gr.stub_params.param_count(3)
	assert_eq(val, gr.stub_params);

func test_param_count_sets_param_count():
	var val = gr.stub_params.param_count(3)
	assert_eq(gr.stub_params.parameter_count, 3)

func test_param_count_default_value():
	assert_eq(gr.stub_params.parameter_count, -1)

func test_param_defaults_returns_self():
	var val = gr.stub_params.param_defaults([])
	assert_eq(val, gr.stub_params)

func test_param_defaults_sets_parameter_count():
	gr.stub_params.param_defaults([1, 2, 3])
	assert_eq(gr.stub_params.parameter_count, 3)

func test_parameter_defaults_is_null_by_default():
	assert_null(gr.stub_params.parameter_defaults)

func test_param_defaults_set_parameter_defaults():
	gr.stub_params.param_defaults([1, 2, 3])
	assert_eq(gr.stub_params.parameter_defaults, [1, 2, 3])

func test_has_param_override_is_false_by_default():
	assert_false(gr.stub_params.has_param_override())

func test_when_param_count_set_has_param_override_is_true():
	gr.stub_params.param_count(3)
	assert_true(gr.stub_params.has_param_override())

# --------------
# Parameter Override Only
# --------------
func test_is_paramter_override_only_false_by_default():
	var sp = StubParamsClass.new()
	assert_false(sp.is_param_override_only())

func test_param_count_override_params_sets_flag():
	var sp = StubParamsClass.new()
	sp.param_count(10)
	assert_true(sp.is_param_override_only())

func test_setting_defaults_sets_flag():
	var sp = StubParamsClass.new()
	sp.param_defaults([1, 2, 3])
	assert_true(sp.is_param_override_only())

func test_to_return_sets_override_flag():
	var sp = StubParamsClass.new()
	sp.param_count(10)
	sp.to_return(7)
	assert_false(sp.is_param_override_only())

func test_order_of_calls_with_to_return_does_not_matter():
	var sp = StubParamsClass.new()
	sp.to_return(7)
	sp.param_count(10)
	assert_false(sp.is_param_override_only())

func test_to_do_nothing_sets_flag():
	var sp = StubParamsClass.new()
	sp.param_count(10)
	sp.to_do_nothing()
	assert_false(sp.is_param_override_only())

func test_to_call_super_sets_flag():
	var sp = StubParamsClass.new()
	sp.param_count(10)
	sp.to_call_super()
	assert_false(sp.is_param_override_only())

# I think this is how it should work.  You may want (if you even can) to
# stub the paramters of a double when it is passed specific values.  In
# all other cases that I can think of, you will end up calling one of the
# other stub methods that flip the flag.
func test_when_passed_does_not_set_flag():
	var sp = StubParamsClass.new()
	sp.param_count(10)
	sp.when_passed(1, 2, 3)
	assert_true(sp.is_param_override_only())


# ------------------------------------------------------------------------------
# Stubbing complex parameter lists
#
# The following tests use the draw_polyline_colors method to test various
# parameter stubbing tasks.
#
# draw_polyline_colors has:
# 	4 parameters, 2 defaults
# 	parameter 3 = -1.0
# 	parameter 4 = false
# ------------------------------------------------------------------------------
func test__draw_polyline_colors__method_meta_1():
	var inst = autofree(Button.new())
	var meta = find_method_meta(inst.get_method_list(), 'draw_polyline_colors')
	var sp = StubParamsClass.new(inst, meta)
	assert_eq(sp.stub_method, meta.name)

func test__draw_polyline_colors__method_meta_2():
	var inst = autofree(Button.new())
	var meta = find_method_meta(inst.get_method_list(), 'draw_polyline_colors')
	var sp = StubParamsClass.new(inst, meta)

	assert_eq(sp.parameter_defaults, [null, null, meta.default_args[0], meta.default_args[1]])
	if(is_failing()):
		print(sp.parameter_defaults)
		meta = find_method_meta(inst.get_method_list(), 'draw_polyline_colors')
		print(meta.default_args)

func test__draw_polyline_colors__method_meta_3():
	var inst = autofree(Button.new())
	var meta = find_method_meta(inst.get_method_list(), 'draw_polyline_colors')
	var sp = StubParamsClass.new(inst, meta)
	assert_true(sp.is_param_override_only())

func test__draw_polyline_colors__method_meta_4():
	var inst = autofree(Button.new())
	var meta = find_method_meta(inst.get_method_list(), 'draw_polyline_colors')
	var sp = StubParamsClass.new(inst, meta)
	assert_eq(sp.parameter_defaults.size(), 4)



# ------------------------------------------------------------------------------
# Test creating from Callable
# ------------------------------------------------------------------------------
func test_can_create_from_callable():
	var sp = StubParamsClass.new(self.assert_true)
	assert_eq(sp.stub_target, self, 'target')
	assert_eq(sp.stub_method, 'assert_true', 'method')

func test_can_create_from_bound_callable():
	var sp = StubParamsClass.new(self.assert_true.bind(false))
	assert_eq(sp.parameters, [false])

func test_when_callable_is_not_bound_parameters_is_null():
	var sp = StubParamsClass.new(self.assert_false)
	assert_eq(sp.parameters, null)


