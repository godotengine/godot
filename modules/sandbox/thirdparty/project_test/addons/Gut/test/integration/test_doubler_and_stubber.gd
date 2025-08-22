extends "res://addons/gut/test.gd"


class BaseTest:
	extends GutTest
	var Stubber = load('res://addons/gut/stubber.gd')
	var Doubler = load('res://addons/gut/doubler.gd')
	var StubParams = load('res://addons/gut/stub_params.gd')
	var Wm = GutUtils.WarningsManager


	const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
	const DOUBLE_ME_SCENE_PATH = 'res://test/resources/doubler_test_objects/double_me_scene.tscn'
	const DOUBLE_EXTENDS_NODE2D = 'res://test/resources/doubler_test_objects/double_extends_node2d.gd'
	const DOUBLE_EXTENDS_WINDOW_DIALOG = 'res://test/resources/doubler_test_objects/double_extends_window_dialog.gd'
	const TO_STUB_PATH = 'res://test/resources/stub_test_objects/to_stub.gd'
	const DOUBLE_WITH_STATIC = 'res://test/resources/doubler_test_objects/has_static_method.gd'
	const INIT_PARAMETERS = 'res://test/resources/stub_test_objects/init_parameters.gd'
	const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
	const DOUBLE_DEFAULT_PARAMETERS = 'res://test/resources/doubler_test_objects/double_default_parameters.gd'

	var DoubleMe = Wm.load_script_ignoring_all_warnings(DOUBLE_ME_PATH)
	var DoubleExtendsNode2D = Wm.load_script_ignoring_all_warnings(DOUBLE_EXTENDS_NODE2D)
	var DoubleExtendsWindowDialog = Wm.load_script_ignoring_all_warnings(DOUBLE_EXTENDS_WINDOW_DIALOG)
	var DoubleWithStatic = Wm.load_script_ignoring_all_warnings(DOUBLE_WITH_STATIC)
	var DoubleMeScene = Wm.load_script_ignoring_all_warnings(DOUBLE_ME_SCENE_PATH)
	var InnerClasses = Wm.load_script_ignoring_all_warnings(INNER_CLASSES_PATH)
	var DoubleDefaultParameters = Wm.load_script_ignoring_all_warnings(DOUBLE_DEFAULT_PARAMETERS)
	var InitParameters = Wm.load_script_ignoring_all_warnings(INIT_PARAMETERS)




class TestTheBasics:
	extends BaseTest

	var gr = {
		doubler = null,
		stubber = null
	}


	func before_each():
		gr.doubler = Doubler.new()
		gr.stubber = Stubber.new()
		gr.doubler.set_stubber(gr.stubber)
		gr.doubler.add_ignored_method(DoubleMe, '_notification')


	func test_stubbing_method_returns_expected_value():
		var D = gr.doubler.double(DoubleMe)
		var sp = StubParams.new(DOUBLE_ME_PATH, 'get_value').to_return(7)
		gr.stubber.add_stub(sp)
		var dbl = autofree(D.new())
		assert_is(dbl, DoubleMe, 'it is a DoubleMe')
		assert_eq(dbl.get_value(), 7)
		if(is_failing()):
			print(DoubleMe, D, dbl)
			# print(gr.stubber.to_s())

	func test_can_stub_non_local_methods():
		var D = autofree(gr.doubler.double(DoubleMe))
		var sp = StubParams.new(DOUBLE_ME_PATH, 'get_position').to_return(Vector2(11, 11))
		gr.stubber.add_stub(sp)
		assert_eq(autofree(D.new()).get_position(), Vector2(11, 11))

	func test_when_non_local_methods_not_stubbed_super_is_returned():
		var D = autofree(gr.doubler.double(DoubleMe))
		var d = autofree(D.new())
		assert_eq(d.get_child_count(), 0)

	func test_can_stub_doubled_instance_values():
		var D = autofree(gr.doubler.double(DoubleMe))
		var d1 = autofree(D.new())
		var d2 = autofree(D.new())

		var sp1 = StubParams.new(DOUBLE_ME_PATH, 'get_value').to_return(5)
		gr.stubber.add_stub(sp1)
		var sp2 = StubParams.new(d1, 'get_value').to_return(10)
		gr.stubber.add_stub(sp2)

		assert_eq(d1.get_value(), 10, 'instantiate gets right value')
		assert_eq(d2.get_value(), 5, 'other instantiate gets class value')

	func test_stubbed_methods_send_parameters_in_callback():
		var sp = StubParams.new(DOUBLE_ME_PATH, 'has_one_param')
		sp.to_return(10).when_passed(1)
		gr.stubber.add_stub(sp)
		var d = autofree(gr.doubler.double(DoubleMe).new())
		assert_eq(d.has_one_param(1), 10)
		assert_eq(d.has_one_param('asdf'), null)

	func test_stub_with_nothing_works_with_parameters():
		var sp1 = StubParams.new(DOUBLE_ME_PATH, 'has_one_param').to_return(5)
		var sp2 = StubParams.new(DOUBLE_ME_PATH, 'has_one_param')
		sp2.to_return(10).when_passed(1)
		gr.stubber.add_stub(sp1)
		gr.stubber.add_stub(sp2)

		var d = autofree(gr.doubler.double(DoubleMe).new())
		assert_eq(d.has_one_param(), 5)
		if(is_failing()):
			print(gr.stubber.to_s())

	func test_can_stub_doubled_scenes():
		var sp = StubParams.new(DOUBLE_ME_SCENE_PATH, 'return_hello')
		sp.to_return('world')
		gr.stubber.add_stub(sp)
		var inst = autofree(gr.doubler.double_scene(DoubleMeScene).instantiate())
		assert_eq(inst.return_hello(), 'world')

	func test_when_stubbed_to_call_super_then_super_is_called():
		var doubled = autofree(gr.doubler.double(DoubleMe).new())
		var params = GutUtils.StubParams.new(doubled, 'set_value').to_call_super()
		gr.stubber.add_stub(params)
		doubled.set_value(99)
		assert_eq(doubled._value, 99)

	func test_when_super_awaits_the_method_awaits():
		var doubled = add_child_autofree(gr.doubler.double(DoubleMe).new())
		var params = GutUtils.StubParams.new(doubled.await_seconds).to_call_super()
		gr.stubber.add_stub(params)
		var before = Time.get_ticks_msec()
		await doubled.await_seconds(1)
		var elapsed = Time.get_ticks_msec() - before
		assert_almost_eq(elapsed, 1000, 300) # 300 seems like a lot, but i guess it's not.

	func test_can_stub_native_methods():
		var d_node2d = autofree(gr.doubler.double_gdnative(Node2D).new())
		var params = GutUtils.StubParams.new(d_node2d, 'get_position').to_return(-1)
		gr.stubber.add_stub(params)
		assert_eq(d_node2d.get_position(), -1)

	func test_partial_double_of_Node2D_returns_super_values():
		var pd_node_2d  = autofree(gr.doubler.partial_double_gdnative(Node2D).new())
		assert_eq(pd_node_2d.is_blocking_signals(), false)

	func test_can_stub_all_Node2D_doubles():
		var d_node2d = autofree(gr.doubler.double_gdnative(Node2D).new())
		var params = GutUtils.StubParams.new(Node2D, 'get_position').to_return(-1)
		gr.stubber.add_stub(params)
		assert_eq(d_node2d.get_position(), -1)
		if(is_failing()):
			print("Node2D = ", Node2D)
			# print(gr.stubber.to_s())

	func test_can_stub_param_count_on_gdnatives():
		var params = GutUtils.StubParams.new(Node2D, 'rpc_id').param_count(5)
		gr.stubber.add_stub(params)
		var d_node2d = autofree(gr.doubler.double_gdnative(Node2D).new())
		assert_eq(gr.stubber.get_parameter_count(d_node2d, 'rpc_id'), 5)


	func test_double_can_have_default_param_values_stubbed():
		var params = GutUtils.StubParams.new(INIT_PARAMETERS, '_init')
		params.param_defaults(["override_default"])
		gr.stubber.add_stub(params)
		var inst = gr.doubler.double(InitParameters).new()
		assert_eq(inst.value, 'override_default')
		if(is_failing()):
			print(gr.stubber.to_s())

	func test_double_can_have_default_param_values_stubbed_after_double_created():
		var Dbl = gr.doubler.double(InitParameters)
		var params = GutUtils.StubParams.new(INIT_PARAMETERS, '_init')
		params.param_defaults(["override_default"])
		gr.stubber.add_stub(params)
		var inst = Dbl.new()
		assert_eq(inst.value, 'override_default')
		if(is_failing()):
			print(gr.stubber.to_s())



class TestInnerClasses:
	extends BaseTest

	var doubler = null
	var stubber = null

	func before_each():
		doubler = Doubler.new()
		stubber = GutUtils.Stubber.new()
		doubler.set_stubber(stubber)


	func test_can_stub_inner_using_loaded_inner_class():
		doubler.inner_class_registry.register(InnerClasses)
		var sp = StubParams.new(InnerClasses.InnerA, 'get_a').to_return(5)
		stubber.add_stub(sp)
		var dbl_inner_a = doubler.double(InnerClasses.InnerA).new()
		assert_eq(stubber.get_return(dbl_inner_a, 'get_a'), 5)



# Since defaults are only available for built-in methods these tests verify
# specific method parameters that were found to cause a problem.
class TestDefaultParameters:
	extends BaseTest

	var doubler = null
	var stubber = null

	func before_each():
		doubler = Doubler.new(GutUtils.DOUBLE_STRATEGY.INCLUDE_NATIVE)
		stubber = GutUtils.Stubber.new()
		doubler.set_stubber(stubber)


	func test_default_values_are_set_in_stubber():
		var dbl = autofree(doubler.double(DoubleDefaultParameters).new())
		var default_value = stubber.get_default_value(DoubleDefaultParameters, 'default_string', 0)
		assert_eq(default_value, 's')
		if(is_failing()):
			print(stubber.to_s())

	func test_partial_gets_deault_values():
		var dbl = autofree(doubler.partial_double(DoubleDefaultParameters).new())
		var result = dbl.return_passed()
		assert_eq(result, 'ab', 'the defauts are a and b')

	func test_partial_gets_passed_values_and_defaults():
		var dbl = autofree(doubler.partial_double(DoubleDefaultParameters).new())
		var result = dbl.return_passed('foo')
		assert_eq(result, 'foob')

	func test_partial_gets_all_values_passed():
		var dbl = autofree(doubler.partial_double(DoubleDefaultParameters).new())
		var result = dbl.return_passed('foo', 'bar')
		assert_eq(result, 'foobar')


class TestMonkeyPatching:
	extends BaseTest

	var doubler = null
	var stubber = null


	var call_this_value = null
	func call_this(value):
		call_this_value = value

	var return_this_value = 99
	func return_this():
		return return_this_value

	func return_passed(p1=null, p2=null, p3=null, p4=null, p5=null, p6=null):
		return [p1, p2, p3, p4, p5, p6]

	func before_each():
		call_this_value = null

		doubler = Doubler.new()
		stubber = GutUtils.Stubber.new()
		doubler.set_stubber(stubber)


	func test_stubbed_method_calls_method():
		var dbl = autofree(doubler.double(DoubleMe).new())
		var params = GutUtils.StubParams.new(dbl.set_value).to_call(call_this)
		stubber.add_stub(params)
		dbl.set_value(9)
		assert_eq(call_this_value, 9)


	func test_stubbed_method_returns_value():
		var dbl = autofree(doubler.double(DoubleMe).new())
		var params = GutUtils.StubParams.new(dbl.get_value).to_call(return_this)
		stubber.add_stub(params)
		var result = dbl.get_value()
		assert_eq(result, 99)


	func test_with_bound_parameters():
		var dbl = autofree(doubler.double(DoubleMe).new())
		var params = GutUtils.StubParams.new(dbl.has_two_params_one_default)
		params.to_call(return_passed.bind('three', 'four'))
		stubber.add_stub(params)
		var result = dbl.has_two_params_one_default('one', 'two')
		assert_eq(result, ["one", "two", "three", "four", null, null])


	func test_with_lambda_that_awaits():
		var lambda = func(p1, _p2=null):
			await get_tree().create_timer(p1).timeout
		var dbl = autofree(doubler.double(DoubleMe).new())

		var params = GutUtils.StubParams.new(dbl.has_two_params_one_default)
		params.to_call(lambda)
		stubber.add_stub(params)

		var before = Time.get_ticks_msec()
		await dbl.has_two_params_one_default(1, 2)
		var elapsed = Time.get_ticks_msec() - before

		assert_almost_eq(elapsed, 1000, 300) # yea, 300 is what you need.


