extends 'res://addons/gut/test.gd'

func test_pass():
	assert_eq('one', 'one')

func test_fail():
	assert_eq(1, 'two')

func test_pending():
	pending('this has text')

