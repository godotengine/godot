extends GutInternalTester

var ParameterHandler = load('res://addons/gut/parameter_handler.gd')

func test_can_make_one():
	assert_not_null(ParameterHandler.new())

func test_can_get_parsed_parameters():
	var ph = ParameterHandler.new([1, 2, 3])
	var results = ph.next_parameters()
	assert_eq(results, 1)

func test_subsequent_calls_moves_through_array():
	var ph = ParameterHandler.new([1, 2, 3])
	var results = ph.next_parameters()
	results = ph.next_parameters()
	results = ph.next_parameters()
	assert_eq(results, 3)

func test_is_done_is_false_by_default():
	var ph = ParameterHandler.new([1, 2, 3])
	assert_false(ph.is_done())

func test_is_done_is_false_when_parameters_remain():
	var ph = ParameterHandler.new([1, 2, 3])
	var results = ph.next_parameters()
	results = ph.next_parameters()
	assert_false(ph.is_done())

func test_is_done_is_true_when_parameters_exhaused():
	var ph = ParameterHandler.new([1, 2, 3])
	var results = ph.next_parameters()
	results = ph.next_parameters()
	results = ph.next_parameters()
	assert_true(ph.is_done())

func test_has_logger():
	assert_has_logger(ParameterHandler.new([]))

func test_passing_non_array_to_constructor_causes_error():
	var ph = ParameterHandler.new('asdf')
	assert_errored(ph, 1)

func test_when_invalid_constructor_parameter_object_is_setup_correctly():
	var ph = ParameterHandler.new('asdf')
	assert_null(ph._params)
	assert_true(ph.is_done(), 'is_done should be true')


