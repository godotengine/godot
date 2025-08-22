extends GutInternalTester


class TestBoth:
	extends GutInternalTester

	const INIT_PARAMETERS = 'res://test/resources/stub_test_objects/init_parameters.gd'
	var InitParameters = load(INIT_PARAMETERS)

	var Doubler = load('res://addons/gut/doubler.gd')

	var _spy = null
	var _doubler = null

	func before_each():
		_spy = Spy.new()
		_doubler = Doubler.new()
		_doubler.set_spy(_spy)

	func after_each():
		_spy.clear()

	func test_spy_is_set_in_metadata():
		var inst = autofree(_doubler.double(DoubleMe).new())
		assert_eq(inst.__gutdbl.spy, _spy)

	func test_when_doubled_method_called_spy_sees_it():
		var inst = autofree(_doubler.double(DoubleMe).new())
		inst.set_value(5)
		assert_true(_spy.was_called(inst, 'set_value'))

	func test_when_doubled_method_called_it_sends_parameters():
		var inst = autofree(_doubler.double(DoubleMe).new())
		inst.set_value(5)
		assert_true(_spy.was_called(inst, 'set_value', [5]))

	func test_it_works_with_two_parameters_too():
		var inst = autofree(_doubler.double(DoubleMe).new())
		inst.has_two_params_one_default('a', 'b')
		assert_false(_spy.was_called(inst, 'has_two_params_one_default', ['c', 'd']), 'should not match')
		assert_true(_spy.was_called(inst, 'has_two_params_one_default', ['a', 'b']), 'should match')

	func test_can_spy_on_built_ins_when_doing_a_full_double():
		_doubler.set_strategy(DOUBLE_STRATEGY.INCLUDE_NATIVE)
		var inst = autofree(_doubler.double(DoubleMe).new())
		# add_user_signal is a function on Object that isn't in our subclass.
		inst.add_user_signal('new_signal', [])
		inst.add_user_signal('signal_with_params', ['a', 'b'])
		assert_true(_spy.was_called(inst, 'add_user_signal'), 'added first signal')
		assert_true(_spy.was_called(inst, 'add_user_signal', ['signal_with_params', ['a', 'b']]), 'second signal added')

	func test_can_spy_on_native_doubles():
		var inst  = autofree(_doubler.partial_double_gdnative(Node2D).new())
		inst.set_position(Vector2(20, 20))
		assert_true(_spy.was_called(inst, 'set_position'))
		assert_true(_spy.was_called(inst, 'set_position', [Vector2(20, 20)]))

	func test_can_spy_on_singleton_doubles():
		pending('No singleton doubling')
		return

		var inst  = _doubler.partial_double_singleton("Input").new()
		inst.is_action_just_pressed("foobar")
		assert_true(_spy.was_called(inst, "is_action_just_pressed"))


	func test_can_spy_on_init():
		var inst = _doubler.double(InitParameters).new('test_value')
		assert_true(_spy.was_called(inst, '_init'))

	func test_can_spy_on_init_parameters():
		var inst = _doubler.double(InitParameters).new('test_value')
		assert_true(_spy.was_called(inst, '_init', ['test_value']))
