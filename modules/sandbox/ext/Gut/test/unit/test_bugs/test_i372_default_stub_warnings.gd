extends GutInternalTester


const DEFAULT_PARAMS_PATH = 'res://test/resources/doubler_test_objects/double_default_parameters.gd'

var DoubleDefaultParams = GutUtils.WarningsManager.load_script_ignoring_all_warnings(DEFAULT_PARAMS_PATH)

func test_for_warnings():
	var Dbl = partial_double(DoubleDefaultParams)
	var inst = Dbl.new()
	var start_warn_count = gut.logger.get_warnings().size()

	stub(inst, 'call_me').param_defaults([null, 'bar'])
	print('******** asserting *************')
	assert_eq(inst.call_call_me('foo'), 'called with foo, bar')
	assert_eq(gut.logger.get_warnings().size(), start_warn_count, 'no warnings')
