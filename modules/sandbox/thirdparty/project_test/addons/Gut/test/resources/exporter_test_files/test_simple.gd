extends 'res://addons/gut/test.gd'

func test_pass_1():
	assert_eq('one', 'one')

func test_pass_2():
	assert_eq('two', 'two')

func test_fail_1():
	assert_eq(1, 'two')

func test_fail_2():
	assert_eq('two', 2)

func test_pending_no_text():
	pending()

func test_pending_with_text():
	pending('this has text')

func test_parameterized_passing(p=use_parameters([1, 2, 3, 4])):
	assert_gt(p, 0)

func test_parameterized_failing(p = use_parameters([[1, 2], [3, 4]])):
	assert_eq(p[0], p[1])
