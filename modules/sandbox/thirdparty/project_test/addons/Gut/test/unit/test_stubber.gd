extends GutInternalTester

var Stubber = load('res://addons/gut/stubber.gd')
# test.gd has a StubParams variable already so this has to have a
# different name.  I thought it was too vague to just use the one
# that test.gd has
var StubParamsClass = load('res://addons/gut/stub_params.gd')

const TO_STUB_PATH = 'res://test/resources/stub_test_objects/to_stub.gd'
var ToStub = load(TO_STUB_PATH)

const HAS_STUB_METADATA_PATH = 'res://test/resources/stub_test_objects/has_stub_metadata.gd'
var HasStubMetadata = load(HAS_STUB_METADATA_PATH)




# The set_return method on Stubber was only used in tests.  I got lazy and
# made this to circumvent it being removed.
class HackedStubber:
	extends 'res://addons/gut/stubber.gd'
	var StubParams = load('res://addons/gut/stub_params.gd')

	func set_return(obj, method, value):
		var sp = StubParams.new(obj, method)
		sp.to_return(value)
		add_stub(sp)
		return sp


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
	stubber = null
}

func print_info(c):
	print('---------')
	for i in range(c.get_method_list().size()):
		print(i, '.  ', c.get_method_list()[i]['name'])
	for i in range(c.get_property_list().size()):
		print(i, '.  ', c.get_property_list()[i], ' = ', c.get(c.get_property_list()[i]['name']))
	print('path = ', c.resource_path)
	print('source = ', c.get_path())
	print('meta = ', c.get_meta_list())
	print('class = ', c.get_class())
	print('script of inst = ', c.new().get_script().get_path())

func before_each():
	gr.stubber = HackedStubber.new()

func test_has_logger():
	assert_has_logger(gr.stubber)

func test_can_get_return_value():
	gr.stubber.set_return(DoubleMe, 'some_method', 7)
	var value = gr.stubber.get_return(DoubleMe, 'some_method')
	assert_eq(value, 7)

func test_can_get_return_for_multiple_methods():
	gr.stubber.set_return(DoubleMe, 'method1', 1)
	gr.stubber.set_return(DoubleMe, 'method2', 2)
	assert_eq(gr.stubber.get_return(DoubleMe, 'method1'), 1, 'method1 returns 1')
	assert_eq(gr.stubber.get_return(DoubleMe, 'method2'), 2, 'method1 returns 2')

func test_can_stub_return_with_class():
	gr.stubber.set_return(ToStub, 'get_value', 0)
	assert_eq(gr.stubber.get_return(ToStub, 'get_value'), 0)

func test_getting_return_for_thing_that_does_not_exist_returns_null():
	var value = gr.stubber.get_return('nothing', 'something')
	assert_eq(value, null)

func test_can_call_super_for_dne_generates_info():
	var value = gr.stubber.should_call_super('nothing', 'something')
	assert_eq(gr.stubber.get_logger().get_infos().size(), 1)


func test_can_get_return_value_using_an_instance_of_class():
	gr.stubber.set_return(ToStub, 'get_value', 0)
	var inst = ToStub.new()
	var value = gr.stubber.get_return(inst, 'get_value')
	assert_eq(value, 0)

func test_instance_stub_takes_precedence_over_path_stub():
	gr.stubber.set_return(TO_STUB_PATH, 'get_value', 0)
	var inst = ToStub.new()
	gr.stubber.set_return(inst, 'get_value', 100)
	var value = gr.stubber.get_return(inst, 'get_value')
	assert_eq(value, 100)

func test_returns_can_be_layered():
	gr.stubber.set_return(ToStub, 'get_value', 0)
	var inst = ToStub.new()
	gr.stubber.set_return(inst, 'get_other', 100)
	assert_eq(gr.stubber.get_return(inst, 'get_value'), 0, 'unstubbed instance method should get class value')
	assert_eq(gr.stubber.get_return(inst, 'get_other'), 100, 'stubbed instance method should get inst value')
	assert_eq(gr.stubber.get_return(ToStub, 'get_value'), 0, 'stubbed path method should get path value')
	assert_eq(gr.stubber.get_return(ToStub ,'get_other'), null, 'unstubbed path method should get null')


func test_will_use_instance_instead_of_metadata():
	gr.stubber.set_return(DoubleMe, 'some_method', 0)
	var inst = HasStubMetadata.new()
	inst.__gutdbl.thepath = DoubleMe
	gr.stubber.set_return(inst, 'some_method', 100)
	assert_eq(gr.stubber.get_return(inst, 'some_method'), 100)

func test_can_stub_with_parameters():
	var sp = gr.stubber.set_return(DoubleMe, 'some_method', 7)
	sp.when_passed(1, 2)
	var val = gr.stubber.get_return(DoubleMe, 'some_method', [1, 2])
	assert_eq(val, 7)

func test_parameter_stubs_return_different_values():
	gr.stubber.set_return(DoubleMe, 'some_method', 5)
	var sp = gr.stubber.set_return(DoubleMe, 'some_method', 10)
	sp.when_passed(1, 2)
	var with_params = gr.stubber.get_return(DoubleMe, 'some_method', [1, 2])
	var wo_params = gr.stubber.get_return(DoubleMe, 'some_method')
	assert_eq(with_params, 10, 'With params should give correct value')
	assert_eq(wo_params, 5, 'Without params should give correct value')

func test_stub_with_nothing_works_with_no_parameters():
	gr.stubber.set_return(DoubleMe, 'has_one_param', 5)
	var sp = gr.stubber.set_return(DoubleMe, 'has_one_param', 10)
	sp.when_passed(1)
	assert_eq(gr.stubber.get_return(DoubleMe, 'has_one_param'), 5)

func test_withStubParams_can_set_return():
	var sp = StubParamsClass.new(DoubleMe, 'method').to_return(10)
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_return(DoubleMe, 'method'), 10)

func test_withStubParams_can_get_return_based_on_parameters():
	var sp = StubParamsClass.new(DoubleMe, 'method').to_return(10).when_passed('a')
	gr.stubber.add_stub(sp)
	var with_params = gr.stubber.get_return(DoubleMe, 'method', ['a'])
	assert_eq(with_params, 10)

func test_withStubParams_can_get_return_based_on_complex_parameters():
	var sp = StubParamsClass.new(DoubleMe, 'method').to_return(10)
	sp.when_passed('a', 1, ['a', 1], sp)
	gr.stubber.add_stub(sp)
	var with_params = gr.stubber.get_return(DoubleMe, 'method', ['a', 1, ['a', 1], sp])
	assert_eq(with_params, 10)

func test_when_parameters_do_not_match_any_stub_then_info_generated():
	var sp = StubParamsClass.new(DoubleMe, 'method').to_return(10).when_passed('a')
	gr.stubber.add_stub(sp)
	gr.stubber.get_return(DoubleMe, 'method', ['b'])
	assert_eq(gr.stubber.get_logger().get_infos().size(), 1)

func test_withStubParams_param_layering_works():
	var sp1 = StubParamsClass.new(DoubleMe, 'method').to_return(10).when_passed(10)
	var sp2 = StubParamsClass.new(DoubleMe, 'method').to_return(5).when_passed(5)
	var sp3 = StubParamsClass.new(DoubleMe, 'method').to_return('nothing')

	gr.stubber.add_stub(sp1)
	gr.stubber.add_stub(sp2)
	gr.stubber.add_stub(sp3)

	var sp1_r = gr.stubber.get_return(DoubleMe, 'method', [10])
	var sp2_r = gr.stubber.get_return(DoubleMe, 'method', [5])
	var sp3_r = gr.stubber.get_return(DoubleMe, 'method')

	assert_eq(sp1_r, 10, 'When passed 10 it gets 10')
	assert_eq(sp2_r, 5, 'When passed 5 it gets 5')
	assert_eq(sp3_r, 'nothing', 'When params do not match it sends default back.')

func test_should_call_super_returns_false_by_default():
	assert_false(gr.stubber.should_call_super('thing', 'method'))

func test_should_call_super_returns_true_when_stubbed_to_do_so():
	var sp = StubParamsClass.new(ToStub, 'method').to_call_super()
	gr.stubber.add_stub(sp)
	var inst = ToStub.new()
	assert_true(gr.stubber.should_call_super(inst, 'method'))

func test_should_call_super_overriden_by_setting_return():
	var sp = StubParamsClass.new(ToStub, 'method').to_call_super()
	sp.to_return(null)
	gr.stubber.add_stub(sp)
	assert_false(gr.stubber.should_call_super(ToStub.new(), 'method'))

func test_when_inner_class_stubbed_instances_of_other_inner_classes_are_not_stubbed():
	var sp = StubParamsClass.new(InnerClasses.InnerA, 'get_a')
	sp.to_return(5)
	gr.stubber.add_stub(sp)

	var another_a = InnerClasses.AnotherInnerA.new()
	var inner_a = InnerClasses.InnerA.new()
	assert_null(gr.stubber.get_return(another_a, 'get_a'), 'AnotherInnerA not stubbed')
	assert_eq(gr.stubber.get_return(inner_a, 'get_a'), 5, 'InnerA is stubbed')

func test_when_instances_of_inner_classes_are_stubbed_only_the_stubbed_instance_is_found():
	var inner_a = InnerClasses.InnerA.new()
	var another_a = InnerClasses.AnotherInnerA.new()

	var sp = StubParamsClass.new(inner_a, 'get_a').to_return('foo')
	gr.stubber.add_stub(sp)
	assert_null(gr.stubber.get_return(another_a, 'get_a'), 'AnotherInnerA not stubbed')
	assert_eq(gr.stubber.get_return(inner_a, 'get_a'), 'foo', 'InnerA is stubbed')

func test_get_call_this_returns_null_by_default():
	assert_null(gr.stubber.get_call_this('thing', 'method'))

func test_get_call_this_returns_method_on_match():
	var call_this = func(): print('hello')
	var sp = StubParamsClass.new(ToStub, 'method').to_call(call_this)
	gr.stubber.add_stub(sp)
	var inst = ToStub.new()
	assert_eq(gr.stubber.get_call_this(inst, 'method'), call_this)

# ----------------
# Parameter Count
# ----------------
func test_get_parameter_count_returns_null_by_default():
	assert_null(gr.stubber.get_parameter_count(DoubleMe, 'method'))

func test_get_parameter_count_returns_stub_params_value():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_count(3)
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_parameter_count(DoubleMe, 'method'), 3)

func test_get_parameter_count_returns_null_when_param_count_not_set():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	gr.stubber.add_stub(sp)
	assert_null(gr.stubber.get_parameter_count(DoubleMe, 'method'))

func test_get_parameter_count_finds_count_when_another_stub_exists():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_count(3)
	gr.stubber.add_stub(sp)

	var second_sp = StubParamsClass.new(DoubleMe, 'method')
	second_sp.to_call_super()
	gr.stubber.add_stub(second_sp)

	assert_eq(gr.stubber.get_parameter_count(DoubleMe, 'method'), 3)

func test_can_stub_parameter_count_for_gdnatives():
	var sp = StubParamsClass.new(Node, 'rpc_id').param_count(5)
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_parameter_count(Node, 'rpc_id'), 5)

func test_can_get_parameter_count_from_instance_of_gdnatives():
	var sp = StubParamsClass.new(Node, 'rpc_id').param_count(5)
	gr.stubber.add_stub(sp)
	var n = double(Node).new()
	assert_eq(gr.stubber.get_parameter_count(n, 'rpc_id'), 5)



# ----------------
# Default Parameter Values
# ----------------
func test_get_default_value_returns_null_by_default():
	assert_null(gr.stubber.get_default_value(DoubleMe, 'method', 0))

func test_get_default_value_returns_stub_param_value_for_index():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_defaults([1, 2, 3])
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_default_value(DoubleMe, 'method', 1), 2)

func test_get_default_value_returns_null_when_only_count_has_been_set():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_count(3)
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_default_value(DoubleMe, 'method', 1), null)

func test_get_default_value_returns_null_when_index_outside_of_range():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_defaults([1, 2, 3])
	gr.stubber.add_stub(sp)
	assert_eq(gr.stubber.get_default_value(DoubleMe, 'method', 99), null)

func test_get_default_values_finds_values_when_another_stub_exists():
	var sp = StubParamsClass.new(DoubleMe, 'method')
	sp.param_defaults([1, 2, 3])
	gr.stubber.add_stub(sp)

	var second_sp = StubParamsClass.new(DoubleMe, 'method')
	second_sp.to_call_super()
	gr.stubber.add_stub(second_sp)

	assert_eq(gr.stubber.get_default_value(DoubleMe, 'method', 1), 2)


# ----------------
# Default Parameter Values from meta
# ----------------
func test_draw_parameter_method_meta():
	# 5 parameters, 2 defaults
	# index 3 = null object
	# index 4 = 1
	var meta = find_method_meta(ToStub.get_script_method_list(), 'default_value_method')
	gr.stubber.stub_defaults_from_meta(ToStub, meta)
	assert_eq(gr.stubber.get_default_value(ToStub, 'default_value_method', 0), 'a')

