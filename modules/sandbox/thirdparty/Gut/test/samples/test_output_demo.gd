extends 'res://addons/gut/test.gd'

func test_passing():
	assert_eq('one', 'one')

func test_failing():
	assert_eq(1, 'two')

func test_pending_no_text():
	pending()

func test_pending_with_text():
	pending('this has text')

func test_parameterized_passing(p=use_parameters([1, 2, 3, 4])):
	assert_gt(p, 0)

func test_parameterized_failing(p = use_parameters([[1, 2], [3, 4]])):
	assert_eq(p[0], p[1])

func test_assert_called_failing():
	var d = double(Node2D).new()
	d.set_position(Vector2(1, 1))
	d.get_name()
	assert_called(d, 'get_position')
